# Copyright (c) 2025, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Output Triton code from public Helion kernels.")
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile", required=True)
    args = parser.parse_args()

    try:
        import helion
        from helion.runtime.kernel import Kernel

        compiled_kernels: list[tuple[Kernel, object, str]] = []
        
        # Patch kernel decorator to set autotune_effort='none' by default
        original_kernel = helion.kernel
        
        def patched_kernel(*args, **kwargs):
            if 'config' not in kwargs and 'autotune_effort' not in kwargs:
                kwargs['autotune_effort'] = 'none'
            return original_kernel(*args, **kwargs)
        
        helion.kernel = patched_kernel
        
        original_call = Kernel.__call__

        def patched_call(self, *call_args, **call_kwargs):
            result = original_call(self, *call_args, **call_kwargs)
            
            try:
                bound = self.bind(call_args)
                cfg = bound.config_spec.default_config()
                triton_code = bound.to_triton_code(cfg)
                compiled_kernels.append((self, call_args, triton_code))
            except Exception:
                pass
            
            return result

        Kernel.__call__ = patched_call

        spec = importlib.util.spec_from_file_location("example", args.inputfile)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        Kernel.__call__ = original_call

        with open(args.outputfile, "w", encoding="utf-8") as out:
            for kernel, args_used, triton_code in compiled_kernels:
                out.write(triton_code)
                out.write("\n\n")

    except Exception as error:
        messages = [m for m in (getattr(error, "args", None) or [str(error)])]
        with contextlib.suppress(Exception):
            sys.stderr.writelines([str(m) + "\n" for m in messages])
        sys.exit(255)


if __name__ == "__main__":
    main()


