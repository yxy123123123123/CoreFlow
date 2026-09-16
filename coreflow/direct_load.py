from __future__ import annotations
import types
from pathlib import Path
import torch
from .io import canonical_module_name,load_json,sha256_file
from .loraflow import activate_official_pythonpath,patch_lora_num,transplant_gate
from .runtime import coreflow_delta,packed_delta

def load_base_with_gate(model_path,official_root,gate_path,expert_order,dtype=torch.bfloat16,device='cpu'):
    patch=patch_lora_num(official_root,len(expert_order));activate_official_pythonpath(official_root)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True);tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(model_path),torch_dtype=dtype,low_cpu_mem_usage=True,local_files_only=True).to(dtype=dtype)
    for layer in model.model.layers: layer.temperature=1.1
    gate=transplant_gate(model,gate_path,expected_k=len(expert_order));model.eval()
    return tok,model,{'patch':patch,'gate':gate,'source_lora_loaded':False}

def _core_forward(self,x,lora_weights=None):
    result=self._direct_base_forward(x)
    if lora_weights is None: raise ValueError('Direct CoreFlow requires gate weights')
    return (result+coreflow_delta(x,lora_weights,self._coreflow_u,self._coreflow_v,self._coreflow_cores)).to(x.dtype)

def _packed_forward(self,x,lora_weights=None):
    result=self._direct_base_forward(x)
    if lora_weights is None: raise ValueError('Direct compact ISVD requires gate weights')
    return (result+packed_delta(x,lora_weights,self._packed_b,self._packed_v,self._packed_owner)).to(x.dtype)

def install_core_direct(model,bank_dir,expert_order):
    from safetensors import safe_open
    bank_dir=Path(bank_dir);cfg=load_json(bank_dir/'core_config.json')
    if cfg.get('format')!='coreflow-oracle-fix-v1' or list(cfg.get('expert_order',[]))!=list(expert_order): raise ValueError('Core direct bank contract mismatch')
    wanted={x['name']:x for x in cfg['modules']};done=[]
    with safe_open(str(bank_dir/'core_bank.safetensors'),framework='pt',device='cpu') as h:
        for raw,module in model.named_modules():
            name=canonical_module_name(raw)
            if name not in wanted: continue
            if not isinstance(module,torch.nn.Linear): raise TypeError(f'Expected raw Linear at {name}, got {type(module)}')
            item=wanted[name];prefix=item['tensor_prefix'];dtype=module.weight.dtype;device=module.weight.device
            module.register_buffer('_coreflow_u',h.get_tensor(prefix+'.U').to(device=device,dtype=dtype),persistent=False)
            module.register_buffer('_coreflow_v',h.get_tensor(prefix+'.V').to(device=device,dtype=dtype),persistent=False)
            module.register_buffer('_coreflow_cores',h.get_tensor(prefix+'.C').to(device=device,dtype=dtype),persistent=False)
            module._direct_base_forward=module.forward;module.forward=types.MethodType(_core_forward,module);done.append(name)
    missing=sorted(set(wanted)-set(done))
    if missing: raise ValueError(f'Unmatched Core modules: {missing[:5]}')
    return {'runtime_kind':'core_direct','installed_modules':len(done),'bank_sha256':sha256_file(bank_dir/'core_bank.safetensors'),'source_lora_loaded':False}

def install_isvd_compact_direct(model,bank_dir,expert_order):
    from safetensors import safe_open
    bank_dir=Path(bank_dir);cfg=load_json(bank_dir/'isvd_compact_config.json')
    if cfg.get('format')!='coreflow-isvd-compact-v1' or list(cfg.get('expert_order',[]))!=list(expert_order): raise ValueError('ISVD compact direct contract mismatch')
    wanted={x['name']:x for x in cfg['modules']};done=[]
    with safe_open(str(bank_dir/'isvd_compact.safetensors'),framework='pt',device='cpu') as h:
        for raw,module in model.named_modules():
            name=canonical_module_name(raw)
            if name not in wanted: continue
            if not isinstance(module,torch.nn.Linear): raise TypeError(f'Expected raw Linear at {name}, got {type(module)}')
            item=wanted[name];prefix=item['compact_tensor_prefix'];dtype=module.weight.dtype;device=module.weight.device
            module.register_buffer('_packed_b',h.get_tensor(prefix+'.B').to(device=device,dtype=dtype),persistent=False)
            module.register_buffer('_packed_v',h.get_tensor(prefix+'.V').to(device=device,dtype=dtype),persistent=False)
            module.register_buffer('_packed_owner',h.get_tensor(prefix+'.owner').to(device=device),persistent=False)
            module._direct_base_forward=module.forward;module.forward=types.MethodType(_packed_forward,module);done.append(name)
    missing=sorted(set(wanted)-set(done))
    if missing: raise ValueError(f'Unmatched ISVD modules: {missing[:5]}')
    return {'runtime_kind':'isvd_compact_direct','installed_modules':len(done),'bank_sha256':sha256_file(bank_dir/'isvd_compact.safetensors'),'source_lora_loaded':False,
            'compact_parameters':cfg['compact_parameters']}
