# Copyright (c) 2023, Compiler Explorer Authors
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
import argparse
import sys
from numba.core import dispatcher

# TODO(Rupt) notes before I forget:
# - Add signature and name as comments
# - Make filter options work
# - Make translation to intel syntax work
# - Add numba-specific filter options
# - Move exec an writer to their own functions
# - Add tests
# - Allow positional arguments?
# - Inspect other states?
# - Name demangling?
# - Move to disasms?


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output compiled asm from public numba-compiled functions."
    )
    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outputfile")

    args = parser.parse_args()

    writer = (
        sys.stdout
        if args.outputfile is None
        else open(args.outputfile, "w", encoding="utf-8")
    )

    namespace = {}
    exec(open(args.inputfile).read(), namespace)

    for key, value in namespace.items():
        if key.startswith("_") or not isinstance(value, dispatcher.Dispatcher):
            continue
        for signature, asm in value.inspect_asm().items():
            # TODO(Rupt) make a comment?
            del signature  # Unused
            # writer.write(f"{key} {signature}\n")
            writer.write(asm)
            return  # TODO(Rupt) handle more than one


if __name__ == "__main__":
    main()
