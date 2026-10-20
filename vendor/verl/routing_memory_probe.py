"""Opt-in metadata-only routing-memory probe; never clones or retains tensors."""
import functools
import json
import os
import socket
import threading
import time


def tensor_info(tensor):
    # Jagged tensors have symbolic dimensions and no directly addressable storage.
    # Inspect their packed values without materializing padding or cloning data.
    nested = bool(getattr(tensor, 'is_nested', False))
    values = tensor.values() if nested else tensor
    storage = values.untyped_storage()
    result = dict(shape=[d if isinstance(d, int) else str(d) for d in tensor.shape],
                  dtype=str(tensor.dtype), device=str(tensor.device), nested=nested,
                  logical_bytes=values.numel() * values.element_size(),
                  storage_bytes=storage.nbytes(), storage_ptr=storage.data_ptr())
    if nested:
        result['values_shape'] = list(values.shape)
        offsets = tensor.offsets()
        result['offsets_bytes'] = offsets.numel() * offsets.element_size()
    return result


def process_info():
    values = {}
    with open('/proc/self/status') as stream:
        for line in stream:
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem'):
                values[key + '_bytes'] = int(value.split()[0]) * 1024
    return values


def emit(event, **fields):
    row = dict(event=event, timestamp=time.time(), pid=os.getpid(), host="worker",
               run=os.environ.get('EXPERIMENT_ID'), **process_info(), **fields)
    print('ROUTING_MEMORY_PROBE ' + json.dumps(row, sort_keys=True), flush=True)


def report_batch(event, data, **fields):
    if isinstance(data, (tuple, list)):
        for i, item in enumerate(data):
            report_batch(event + '/' + str(i), item, **fields)
        return
    batch = getattr(data, 'batch', None)
    if batch is None:
        return
    tensors = {}
    for k, v in batch.items():
        if k in ('routed_experts', 'responses', 'input_ids', 'attention_mask'):
            try:
                tensors[k] = tensor_info(v)
            except Exception as error:
                tensors[k] = {'measurement_error': str(error), 'type': str(type(v))}
    emit(event, tensors=tensors, **fields)


def install_protocol(module):
    DataProto = module.DataProto
    if getattr(DataProto, '_routing_memory_probe_installed', False):
        return
    DataProto._routing_memory_probe_installed = True
    original_post = DataProto.__post_init__

    def post(self):
        original_post(self)
        if self.batch is not None and 'routed_experts' in self.batch and len(self) >= 32:
            report_batch('dataproto_routes', self)

    DataProto.__post_init__ = post


def install_trainer(module):
    RayPPOTrainer = module.RayPPOTrainer
    for name in ('_compute_old_log_prob', '_update_actor'):
        original = getattr(RayPPOTrainer, name)

        def wrap(fn, stage):
            @functools.wraps(fn)
            def observed(self, batch, *args, **kwargs):
                step = getattr(self, 'global_steps', None)
                report_batch(stage + '/before', batch, step=step)
                start = time.monotonic()
                result = fn(self, batch, *args, **kwargs)
                report_batch(stage + '/after', result, step=step,
                             elapsed_seconds=time.monotonic() - start)
                return result
            return observed
        setattr(RayPPOTrainer, name, wrap(original, name))
    original_fit = RayPPOTrainer.fit

    @functools.wraps(original_fit)
    def fit(self, *args, **kwargs):
        stop = threading.Event()

        def sample():
            while not stop.wait(5):
                emit('trainer_process_sample', step=getattr(self, 'global_steps', None))

        emit('trainer_fit_start')
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            return original_fit(self, *args, **kwargs)
        finally:
            stop.set()
            emit('trainer_fit_end')

    RayPPOTrainer.fit = fit


def setup():
    if os.environ.get('ROUTING_MEMORY_PROBE') != '1':
        return
    # Ray calls setup before assigning per-task CUDA visibility. Do not import
    # torch/verl here: device.py caches availability at import time.
    import sys
    from importlib.machinery import PathFinder
    callbacks = {'verl.protocol': install_protocol,
                 'verl.trainer.ppo.ray_trainer': install_trainer}

    class Loader:
        def __init__(self, original, callback):
            self.original, self.callback = original, callback

        def create_module(self, spec):
            method = getattr(self.original, 'create_module', None)
            return method(spec) if method else None

        def exec_module(self, module):
            self.original.exec_module(module)
            self.callback(module)

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in callbacks:
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader, callbacks.pop(fullname))
            return spec

    for name in tuple(callbacks):
        if name in sys.modules:
            callbacks.pop(name)(sys.modules[name])
    sys.meta_path.insert(0, Finder())
    emit('probe_installed_deferred')
