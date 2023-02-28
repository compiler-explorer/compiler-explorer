import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ASYNC_GEN_WRAP":
            return {
                "html": "Wraps the value on top of the stack in an async_generator_wrapped_value.\nUsed to yield in async generators.\nNew in version 3.11.",
                "tooltip": "Wraps the value on top of the stack in an async_generator_wrapped_value.\nUsed to yield in async generators.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-ASYNC_GEN_WRAP"
            };

        case "BEFORE_ASYNC_WITH":
            return {
                "html": "Resolves __aenter__ and __aexit__ from the object on top of the\nstack.  Pushes __aexit__ and result of __aenter__() to the stack.\nNew in version 3.5.",
                "tooltip": "Resolves __aenter__ and __aexit__ from the object on top of the\nstack.  Pushes __aexit__ and result of __aenter__() to the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BEFORE_ASYNC_WITH"
            };

        case "BEFORE_WITH":
            return {
                "html": "This opcode performs several operations before a with block starts.  First,\nit loads __exit__() from the context manager and pushes it onto\nthe stack for later use by WITH_EXCEPT_START.  Then,\n__enter__() is called. Finally, the result of calling the\n__enter__() method is pushed onto the stack.\nNew in version 3.11.",
                "tooltip": "This opcode performs several operations before a with block starts.  First,\nit loads __exit__() from the context manager and pushes it onto\nthe stack for later use by WITH_EXCEPT_START.  Then,\n__enter__() is called. Finally, the result of calling the\n__enter__() method is pushed onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BEFORE_WITH"
            };

        case "BINARY_OP":
            return {
                "html": "Implements the binary and in-place operators (depending on the value of\nop).\nNew in version 3.11.",
                "tooltip": "Implements the binary and in-place operators (depending on the value of\nop).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BINARY_OP"
            };

        case "BINARY_SUBSCR":
            return {
                "html": "Implements TOS = TOS1[TOS].",
                "tooltip": "Implements TOS = TOS1[TOS].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BINARY_SUBSCR"
            };

        case "BUILD_CONST_KEY_MAP":
            return {
                "html": "The version of BUILD_MAP specialized for constant keys. Pops the\ntop element on the stack which contains a tuple of keys, then starting from\nTOS1, pops count values to form values in the built dictionary.\nNew in version 3.6.",
                "tooltip": "The version of BUILD_MAP specialized for constant keys. Pops the\ntop element on the stack which contains a tuple of keys, then starting from\nTOS1, pops count values to form values in the built dictionary.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_CONST_KEY_MAP"
            };

        case "BUILD_LIST":
            return {
                "html": "Works as BUILD_TUPLE, but creates a list.",
                "tooltip": "Works as BUILD_TUPLE, but creates a list.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_LIST"
            };

        case "BUILD_MAP":
            return {
                "html": "Pushes a new dictionary object onto the stack.  Pops 2 * count items\nso that the dictionary holds count entries:\n{..., TOS3: TOS2, TOS1: TOS}.\nChanged in version 3.5: The dictionary is created from stack items instead of creating an\nempty dictionary pre-sized to hold count items.",
                "tooltip": "Pushes a new dictionary object onto the stack.  Pops 2 * count items\nso that the dictionary holds count entries:\n{..., TOS3: TOS2, TOS1: TOS}.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_MAP"
            };

        case "BUILD_SET":
            return {
                "html": "Works as BUILD_TUPLE, but creates a set.",
                "tooltip": "Works as BUILD_TUPLE, but creates a set.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_SET"
            };

        case "BUILD_SLICE":
            return {
                "html": "Pushes a slice object on the stack.  argc must be 2 or 3.  If it is 2,\nslice(TOS1, TOS) is pushed; if it is 3, slice(TOS2, TOS1, TOS) is\npushed. See the slice() built-in function for more information.",
                "tooltip": "Pushes a slice object on the stack.  argc must be 2 or 3.  If it is 2,\nslice(TOS1, TOS) is pushed; if it is 3, slice(TOS2, TOS1, TOS) is\npushed. See the slice() built-in function for more information.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_SLICE"
            };

        case "BUILD_STRING":
            return {
                "html": "Concatenates count strings from the stack and pushes the resulting string\nonto the stack.\nNew in version 3.6.",
                "tooltip": "Concatenates count strings from the stack and pushes the resulting string\nonto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING"
            };

        case "BUILD_TUPLE":
            return {
                "html": "Creates a tuple consuming count items from the stack, and pushes the\nresulting tuple onto the stack.",
                "tooltip": "Creates a tuple consuming count items from the stack, and pushes the\nresulting tuple onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_TUPLE"
            };

        case "CACHE":
            return {
                "html": "Rather than being an actual instruction, this opcode is used to mark extra\nspace for the interpreter to cache useful data directly in the bytecode\nitself. It is automatically hidden by all dis utilities, but can be\nviewed with show_caches=True.\nLogically, this space is part of the preceding instruction. Many opcodes\nexpect to be followed by an exact number of caches, and will instruct the\ninterpreter to skip over them at runtime.\nPopulated caches can look like arbitrary instructions, so great care should\nbe taken when reading or modifying raw, adaptive bytecode containing\nquickened data.\nNew in version 3.11.",
                "tooltip": "Rather than being an actual instruction, this opcode is used to mark extra\nspace for the interpreter to cache useful data directly in the bytecode\nitself. It is automatically hidden by all dis utilities, but can be\nviewed with show_caches=True.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CACHE"
            };

        case "CALL":
            return {
                "html": "Calls a callable object with the number of arguments specified by argc,\nincluding the named arguments specified by the preceding\nKW_NAMES, if any.\nOn the stack are (in ascending order), either:\nNULL\nThe callable\nThe positional arguments\nThe named arguments\nor:\nThe callable\nself\nThe remaining positional arguments\nThe named arguments\nargc is the total of the positional and named arguments, excluding\nself when a NULL is not present.\nCALL pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.\nNew in version 3.11.",
                "tooltip": "Calls a callable object with the number of arguments specified by argc,\nincluding the named arguments specified by the preceding\nKW_NAMES, if any.\nOn the stack are (in ascending order), either",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL"
            };

        case "CALL_FUNCTION_EX":
            return {
                "html": "Calls a callable object with variable set of positional and keyword\narguments.  If the lowest bit of flags is set, the top of the stack\ncontains a mapping object containing additional keyword arguments.\nBefore the callable is called, the mapping object and iterable object\nare each \u201cunpacked\u201d and their contents passed in as keyword and\npositional arguments respectively.\nCALL_FUNCTION_EX pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.\nNew in version 3.6.",
                "tooltip": "Calls a callable object with variable set of positional and keyword\narguments.  If the lowest bit of flags is set, the top of the stack\ncontains a mapping object containing additional keyword arguments.\nBefore the callable is called, the mapping object and iterable object\nare each \u201cunpacked\u201d and their contents passed in as keyword and\npositional arguments respectively.\nCALL_FUNCTION_EX pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL_FUNCTION_EX"
            };

        case "CHECK_EG_MATCH":
            return {
                "html": "Performs exception matching for except*. Applies split(TOS) on\nthe exception group representing TOS1.\nIn case of a match, pops two items from the stack and pushes the\nnon-matching subgroup (None in case of full match) followed by the\nmatching subgroup. When there is no match, pops one item (the match\ntype) and pushes None.\nNew in version 3.11.",
                "tooltip": "Performs exception matching for except*. Applies split(TOS) on\nthe exception group representing TOS1.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CHECK_EG_MATCH"
            };

        case "CHECK_EXC_MATCH":
            return {
                "html": "Performs exception matching for except. Tests whether the TOS1 is an exception\nmatching TOS. Pops TOS and pushes the boolean result of the test.\nNew in version 3.11.",
                "tooltip": "Performs exception matching for except. Tests whether the TOS1 is an exception\nmatching TOS. Pops TOS and pushes the boolean result of the test.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CHECK_EXC_MATCH"
            };

        case "COMPARE_OP":
            return {
                "html": "Performs a Boolean operation.  The operation name can be found in\ncmp_op[opname].",
                "tooltip": "Performs a Boolean operation.  The operation name can be found in\ncmp_op[opname].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COMPARE_OP"
            };

        case "CONTAINS_OP":
            return {
                "html": "Performs in comparison, or not in if invert is 1.\nNew in version 3.9.",
                "tooltip": "Performs in comparison, or not in if invert is 1.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CONTAINS_OP"
            };

        case "COPY":
            return {
                "html": "Push the i-th item to the top of the stack. The item is not removed from its\noriginal location.\nNew in version 3.11.",
                "tooltip": "Push the i-th item to the top of the stack. The item is not removed from its\noriginal location.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COPY"
            };

        case "COPY_FREE_VARS":
            return {
                "html": "Copies the n free variables from the closure into the frame.\nRemoves the need for special code on the caller\u2019s side when calling\nclosures.\nNew in version 3.11.",
                "tooltip": "Copies the n free variables from the closure into the frame.\nRemoves the need for special code on the caller\u2019s side when calling\nclosures.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COPY_FREE_VARS"
            };

        case "DELETE_ATTR":
            return {
                "html": "Implements del TOS.name, using namei as index into co_names.",
                "tooltip": "Implements del TOS.name, using namei as index into co_names.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_ATTR"
            };

        case "DELETE_DEREF":
            return {
                "html": "Empties the cell contained in slot i of the \u201cfast locals\u201d storage.\nUsed by the del statement.\nNew in version 3.2.\nChanged in version 3.11: i is no longer offset by the length of co_varnames.",
                "tooltip": "Empties the cell contained in slot i of the \u201cfast locals\u201d storage.\nUsed by the del statement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_DEREF"
            };

        case "DELETE_FAST":
            return {
                "html": "Deletes local co_varnames[var_num].",
                "tooltip": "Deletes local co_varnames[var_num].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_FAST"
            };

        case "DELETE_GLOBAL":
            return {
                "html": "Works as DELETE_NAME, but deletes a global name.",
                "tooltip": "Works as DELETE_NAME, but deletes a global name.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_GLOBAL"
            };

        case "DELETE_NAME":
            return {
                "html": "Implements del name, where namei is the index into co_names\nattribute of the code object.",
                "tooltip": "Implements del name, where namei is the index into co_names\nattribute of the code object.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_NAME"
            };

        case "DELETE_SUBSCR":
            return {
                "html": "Implements del TOS1[TOS].",
                "tooltip": "Implements del TOS1[TOS].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_SUBSCR"
            };

        case "DICT_MERGE":
            return {
                "html": "Like DICT_UPDATE but raises an exception for duplicate keys.\nNew in version 3.9.",
                "tooltip": "Like DICT_UPDATE but raises an exception for duplicate keys.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DICT_MERGE"
            };

        case "DICT_UPDATE":
            return {
                "html": "Calls dict.update(TOS1[-i], TOS).  Used to build dicts.\nNew in version 3.9.",
                "tooltip": "Calls dict.update(TOS1[-i], TOS).  Used to build dicts.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DICT_UPDATE"
            };

        case "END_ASYNC_FOR":
            return {
                "html": "Terminates an async for loop.  Handles an exception raised\nwhen awaiting a next item.  If TOS is StopAsyncIteration pop 3\nvalues from the stack and restore the exception state using the second\nof them.  Otherwise re-raise the exception using the value\nfrom the stack.  An exception handler block is removed from the block stack.\nNew in version 3.8: \nChanged in version 3.11: Exception representation on the stack now consist of one, not three, items.",
                "tooltip": "Terminates an async for loop.  Handles an exception raised\nwhen awaiting a next item.  If TOS is StopAsyncIteration pop 3\nvalues from the stack and restore the exception state using the second\nof them.  Otherwise re-raise the exception using the value\nfrom the stack.  An exception handler block is removed from the block stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-END_ASYNC_FOR"
            };

        case "EXTENDED_ARG":
            return {
                "html": "Prefixes any opcode which has an argument too big to fit into the default one\nbyte. ext holds an additional byte which act as higher bits in the argument.\nFor each opcode, at most three prefixal EXTENDED_ARG are allowed, forming\nan argument from two-byte to four-byte.",
                "tooltip": "Prefixes any opcode which has an argument too big to fit into the default one\nbyte. ext holds an additional byte which act as higher bits in the argument.\nFor each opcode, at most three prefixal EXTENDED_ARG are allowed, forming\nan argument from two-byte to four-byte.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-EXTENDED_ARG"
            };

        case "FORMAT_VALUE":
            return {
                "html": "Used for implementing formatted literal strings (f-strings).  Pops\nan optional fmt_spec from the stack, then a required value.\nflags is interpreted as follows:\n(flags & 0x03) == 0x00: value is formatted as-is.\n(flags & 0x03) == 0x01: call str() on value before\nformatting it.\n(flags & 0x03) == 0x02: call repr() on value before\nformatting it.\n(flags & 0x03) == 0x03: call ascii() on value before\nformatting it.\n(flags & 0x04) == 0x04: pop fmt_spec from the stack and use\nit, else use an empty fmt_spec.\nFormatting is performed using PyObject_Format().  The\nresult is pushed on the stack.\nNew in version 3.6.",
                "tooltip": "Used for implementing formatted literal strings (f-strings).  Pops\nan optional fmt_spec from the stack, then a required value.\nflags is interpreted as follows",
                "url": "https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE"
            };

        case "FOR_ITER":
            return {
                "html": "TOS is an iterator.  Call its __next__() method.  If\nthis yields a new value, push it on the stack (leaving the iterator below\nit).  If the iterator indicates it is exhausted, TOS is popped, and the byte\ncode counter is incremented by delta.",
                "tooltip": "TOS is an iterator.  Call its __next__() method.  If\nthis yields a new value, push it on the stack (leaving the iterator below\nit).  If the iterator indicates it is exhausted, TOS is popped, and the byte\ncode counter is incremented by delta.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-FOR_ITER"
            };

        case "GET_AITER":
            return {
                "html": "Implements TOS = TOS.__aiter__().\nNew in version 3.5.\nChanged in version 3.7: Returning awaitable objects from __aiter__ is no longer\nsupported.",
                "tooltip": "Implements TOS = TOS.__aiter__().",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_AITER"
            };

        case "GET_ANEXT":
            return {
                "html": "Pushes get_awaitable(TOS.__anext__()) to the stack.  See\nGET_AWAITABLE for details about get_awaitable.\nNew in version 3.5.",
                "tooltip": "Pushes get_awaitable(TOS.__anext__()) to the stack.  See\nGET_AWAITABLE for details about get_awaitable.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_ANEXT"
            };

        case "GET_AWAITABLE":
            return {
                "html": "Implements TOS = get_awaitable(TOS), where get_awaitable(o)\nreturns o if o is a coroutine object or a generator object with\nthe CO_ITERABLE_COROUTINE flag, or resolves\no.__await__.\nIf the where operand is nonzero, it indicates where the instruction\noccurs:\n1 After a call to __aenter__\n2 After a call to __aexit__\nNew in version 3.5.\nChanged in version 3.11: Previously, this instruction did not have an oparg.",
                "tooltip": "Implements TOS = get_awaitable(TOS), where get_awaitable(o)\nreturns o if o is a coroutine object or a generator object with\nthe CO_ITERABLE_COROUTINE flag, or resolves\no.__await__.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_AWAITABLE"
            };

        case "GET_ITER":
            return {
                "html": "Implements TOS = iter(TOS).",
                "tooltip": "Implements TOS = iter(TOS).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_ITER"
            };

        case "GET_LEN":
            return {
                "html": "Push len(TOS) onto the stack.\nNew in version 3.10.",
                "tooltip": "Push len(TOS) onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_LEN"
            };

        case "GET_YIELD_FROM_ITER":
            return {
                "html": "If TOS is a generator iterator or coroutine object\nit is left as is.  Otherwise, implements TOS = iter(TOS).\nNew in version 3.5.",
                "tooltip": "If TOS is a generator iterator or coroutine object\nit is left as is.  Otherwise, implements TOS = iter(TOS).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_YIELD_FROM_ITER"
            };

        case "HAVE_ARGUMENT":
            return {
                "html": "This is not really an opcode.  It identifies the dividing line between\nopcodes which don\u2019t use their argument and those that do\n(< HAVE_ARGUMENT and >= HAVE_ARGUMENT, respectively).\nChanged in version 3.6: Now every instruction has an argument, but opcodes < HAVE_ARGUMENT\nignore it. Before, only opcodes >= HAVE_ARGUMENT had an argument.",
                "tooltip": "This is not really an opcode.  It identifies the dividing line between\nopcodes which don\u2019t use their argument and those that do\n(< HAVE_ARGUMENT and >= HAVE_ARGUMENT, respectively).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-HAVE_ARGUMENT"
            };

        case "IMPORT_FROM":
            return {
                "html": "Loads the attribute co_names[namei] from the module found in TOS. The\nresulting object is pushed onto the stack, to be subsequently stored by a\nSTORE_FAST instruction.",
                "tooltip": "Loads the attribute co_names[namei] from the module found in TOS. The\nresulting object is pushed onto the stack, to be subsequently stored by a\nSTORE_FAST instruction.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IMPORT_FROM"
            };

        case "IMPORT_NAME":
            return {
                "html": "Imports the module co_names[namei].  TOS and TOS1 are popped and provide\nthe fromlist and level arguments of __import__().  The module\nobject is pushed onto the stack.  The current namespace is not affected: for\na proper import statement, a subsequent STORE_FAST instruction\nmodifies the namespace.",
                "tooltip": "Imports the module co_names[namei].  TOS and TOS1 are popped and provide\nthe fromlist and level arguments of __import__().  The module\nobject is pushed onto the stack.  The current namespace is not affected: for\na proper import statement, a subsequent STORE_FAST instruction\nmodifies the namespace.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IMPORT_NAME"
            };

        case "IMPORT_STAR":
            return {
                "html": "Loads all symbols not starting with '_' directly from the module TOS to\nthe local namespace. The module is popped after loading all names. This\nopcode implements from module import *.",
                "tooltip": "Loads all symbols not starting with '_' directly from the module TOS to\nthe local namespace. The module is popped after loading all names. This\nopcode implements from module import *.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IMPORT_STAR"
            };

        case "IS_OP":
            return {
                "html": "Performs is comparison, or is not if invert is 1.\nNew in version 3.9.",
                "tooltip": "Performs is comparison, or is not if invert is 1.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IS_OP"
            };

        case "JUMP_BACKWARD":
            return {
                "html": "Decrements bytecode counter by delta. Checks for interrupts.\nNew in version 3.11.",
                "tooltip": "Decrements bytecode counter by delta. Checks for interrupts.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_BACKWARD"
            };

        case "JUMP_BACKWARD_NO_INTERRUPT":
            return {
                "html": "Decrements bytecode counter by delta. Does not check for interrupts.\nNew in version 3.11.",
                "tooltip": "Decrements bytecode counter by delta. Does not check for interrupts.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_BACKWARD_NO_INTERRUPT"
            };

        case "JUMP_FORWARD":
            return {
                "html": "Increments bytecode counter by delta.",
                "tooltip": "Increments bytecode counter by delta.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_FORWARD"
            };

        case "JUMP_IF_FALSE_OR_POP":
            return {
                "html": "If TOS is false, increments the bytecode counter by delta and leaves TOS on the\nstack.  Otherwise (TOS is true), TOS is popped.\nNew in version 3.1.\nChanged in version 3.11: The oparg is now a relative delta rather than an absolute target.",
                "tooltip": "If TOS is false, increments the bytecode counter by delta and leaves TOS on the\nstack.  Otherwise (TOS is true), TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_IF_FALSE_OR_POP"
            };

        case "JUMP_IF_TRUE_OR_POP":
            return {
                "html": "If TOS is true, increments the bytecode counter by delta and leaves TOS on the\nstack.  Otherwise (TOS is false), TOS is popped.\nNew in version 3.1.\nChanged in version 3.11: The oparg is now a relative delta rather than an absolute target.",
                "tooltip": "If TOS is true, increments the bytecode counter by delta and leaves TOS on the\nstack.  Otherwise (TOS is false), TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_IF_TRUE_OR_POP"
            };

        case "KW_NAMES":
            return {
                "html": "Prefixes PRECALL.\nStores a reference to co_consts[consti] into an internal variable\nfor use by CALL. co_consts[consti] must be a tuple of strings.\nNew in version 3.11.",
                "tooltip": "Prefixes PRECALL.\nStores a reference to co_consts[consti] into an internal variable\nfor use by CALL. co_consts[consti] must be a tuple of strings.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-KW_NAMES"
            };

        case "LIST_APPEND":
            return {
                "html": "Calls list.append(TOS1[-i], TOS).  Used to implement list comprehensions.",
                "tooltip": "Calls list.append(TOS1[-i], TOS).  Used to implement list comprehensions.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LIST_APPEND"
            };

        case "LIST_EXTEND":
            return {
                "html": "Calls list.extend(TOS1[-i], TOS).  Used to build lists.\nNew in version 3.9.",
                "tooltip": "Calls list.extend(TOS1[-i], TOS).  Used to build lists.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LIST_EXTEND"
            };

        case "LIST_TO_TUPLE":
            return {
                "html": "Pops a list from the stack and pushes a tuple containing the same values.\nNew in version 3.9.",
                "tooltip": "Pops a list from the stack and pushes a tuple containing the same values.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LIST_TO_TUPLE"
            };

        case "LOAD_ASSERTION_ERROR":
            return {
                "html": "Pushes AssertionError onto the stack.  Used by the assert\nstatement.\nNew in version 3.9.",
                "tooltip": "Pushes AssertionError onto the stack.  Used by the assert\nstatement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_ASSERTION_ERROR"
            };

        case "LOAD_ATTR":
            return {
                "html": "Replaces TOS with getattr(TOS, co_names[namei]).",
                "tooltip": "Replaces TOS with getattr(TOS, co_names[namei]).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_ATTR"
            };

        case "LOAD_BUILD_CLASS":
            return {
                "html": "Pushes builtins.__build_class__() onto the stack.  It is later called\nto construct a class.",
                "tooltip": "Pushes builtins.__build_class__() onto the stack.  It is later called\nto construct a class.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_BUILD_CLASS"
            };

        case "LOAD_CLASSDEREF":
            return {
                "html": "Much like LOAD_DEREF but first checks the locals dictionary before\nconsulting the cell.  This is used for loading free variables in class\nbodies.\nNew in version 3.4.\nChanged in version 3.11: i is no longer offset by the length of co_varnames.",
                "tooltip": "Much like LOAD_DEREF but first checks the locals dictionary before\nconsulting the cell.  This is used for loading free variables in class\nbodies.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_CLASSDEREF"
            };

        case "LOAD_CLOSURE":
            return {
                "html": "Pushes a reference to the cell contained in slot i of the \u201cfast locals\u201d\nstorage.  The name of the variable is co_fastlocalnames[i].\nNote that LOAD_CLOSURE is effectively an alias for LOAD_FAST.\nIt exists to keep bytecode a little more readable.\nChanged in version 3.11: i is no longer offset by the length of co_varnames.",
                "tooltip": "Pushes a reference to the cell contained in slot i of the \u201cfast locals\u201d\nstorage.  The name of the variable is co_fastlocalnames[i].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_CLOSURE"
            };

        case "LOAD_CONST":
            return {
                "html": "Pushes co_consts[consti] onto the stack.",
                "tooltip": "Pushes co_consts[consti] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_CONST"
            };

        case "LOAD_DEREF":
            return {
                "html": "Loads the cell contained in slot i of the \u201cfast locals\u201d storage.\nPushes a reference to the object the cell contains on the stack.\nChanged in version 3.11: i is no longer offset by the length of co_varnames.",
                "tooltip": "Loads the cell contained in slot i of the \u201cfast locals\u201d storage.\nPushes a reference to the object the cell contains on the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_DEREF"
            };

        case "LOAD_FAST":
            return {
                "html": "Pushes a reference to the local co_varnames[var_num] onto the stack.",
                "tooltip": "Pushes a reference to the local co_varnames[var_num] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST"
            };

        case "LOAD_GLOBAL":
            return {
                "html": "Loads the global named co_names[namei>>1] onto the stack.\nChanged in version 3.11: If the low bit of namei is set, then a NULL is pushed to the\nstack before the global variable.",
                "tooltip": "Loads the global named co_names[namei>>1] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_GLOBAL"
            };

        case "LOAD_METHOD":
            return {
                "html": "Loads a method named co_names[namei] from the TOS object. TOS is popped.\nThis bytecode distinguishes two cases: if TOS has a method with the correct\nname, the bytecode pushes the unbound method and TOS. TOS will be used as\nthe first argument (self) by CALL when calling the\nunbound method. Otherwise, NULL and the object return by the attribute\nlookup are pushed.\nNew in version 3.7.",
                "tooltip": "Loads a method named co_names[namei] from the TOS object. TOS is popped.\nThis bytecode distinguishes two cases: if TOS has a method with the correct\nname, the bytecode pushes the unbound method and TOS. TOS will be used as\nthe first argument (self) by CALL when calling the\nunbound method. Otherwise, NULL and the object return by the attribute\nlookup are pushed.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_METHOD"
            };

        case "LOAD_NAME":
            return {
                "html": "Pushes the value associated with co_names[namei] onto the stack.",
                "tooltip": "Pushes the value associated with co_names[namei] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_NAME"
            };

        case "MAKE_CELL":
            return {
                "html": "Creates a new cell in slot i.  If that slot is empty then\nthat value is stored into the new cell.\nNew in version 3.11.",
                "tooltip": "Creates a new cell in slot i.  If that slot is empty then\nthat value is stored into the new cell.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAKE_CELL"
            };

        case "MAKE_FUNCTION":
            return {
                "html": "Pushes a new function object on the stack.  From bottom to top, the consumed\nstack must consist of values if the argument carries a specified flag value\n0x01 a tuple of default values for positional-only and\npositional-or-keyword parameters in positional order\n0x02 a dictionary of keyword-only parameters\u2019 default values\n0x04 a tuple of strings containing parameters\u2019 annotations\n0x08 a tuple containing cells for free variables, making a closure\nthe code associated with the function (at TOS1)\nthe qualified name of the function (at TOS)\nChanged in version 3.10: Flag value 0x04 is a tuple of strings instead of dictionary",
                "tooltip": "Pushes a new function object on the stack.  From bottom to top, the consumed\nstack must consist of values if the argument carries a specified flag value",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAKE_FUNCTION"
            };

        case "MAP_ADD":
            return {
                "html": "Calls dict.__setitem__(TOS1[-i], TOS1, TOS).  Used to implement dict\ncomprehensions.\nNew in version 3.1.\nChanged in version 3.8: Map value is TOS and map key is TOS1. Before, those were reversed.",
                "tooltip": "Calls dict.__setitem__(TOS1[-i], TOS1, TOS).  Used to implement dict\ncomprehensions.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAP_ADD"
            };

        case "MATCH_CLASS":
            return {
                "html": "TOS is a tuple of keyword attribute names, TOS1 is the class being matched\nagainst, and TOS2 is the match subject.  count is the number of positional\nsub-patterns.\nPop TOS, TOS1, and TOS2.  If TOS2 is an instance of TOS1 and has the\npositional and keyword attributes required by count and TOS, push a tuple\nof extracted attributes.  Otherwise, push None.\nNew in version 3.10.\nChanged in version 3.11: Previously, this instruction also pushed a boolean value indicating\nsuccess (True) or failure (False).",
                "tooltip": "TOS is a tuple of keyword attribute names, TOS1 is the class being matched\nagainst, and TOS2 is the match subject.  count is the number of positional\nsub-patterns.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_CLASS"
            };

        case "MATCH_KEYS":
            return {
                "html": "TOS is a tuple of mapping keys, and TOS1 is the match subject.  If TOS1\ncontains all of the keys in TOS, push a tuple containing the\ncorresponding values. Otherwise, push None.\nNew in version 3.10.\nChanged in version 3.11: Previously, this instruction also pushed a boolean value indicating\nsuccess (True) or failure (False).",
                "tooltip": "TOS is a tuple of mapping keys, and TOS1 is the match subject.  If TOS1\ncontains all of the keys in TOS, push a tuple containing the\ncorresponding values. Otherwise, push None.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_KEYS"
            };

        case "MATCH_MAPPING":
            return {
                "html": "If TOS is an instance of collections.abc.Mapping (or, more technically: if\nit has the Py_TPFLAGS_MAPPING flag set in its\ntp_flags), push True onto the stack.  Otherwise, push\nFalse.\nNew in version 3.10.",
                "tooltip": "If TOS is an instance of collections.abc.Mapping (or, more technically: if\nit has the Py_TPFLAGS_MAPPING flag set in its\ntp_flags), push True onto the stack.  Otherwise, push\nFalse.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_MAPPING"
            };

        case "MATCH_SEQUENCE":
            return {
                "html": "If TOS is an instance of collections.abc.Sequence and is not an instance\nof str/bytes/bytearray (or, more technically: if it has\nthe Py_TPFLAGS_SEQUENCE flag set in its tp_flags),\npush True onto the stack.  Otherwise, push False.\nNew in version 3.10.",
                "tooltip": "If TOS is an instance of collections.abc.Sequence and is not an instance\nof str/bytes/bytearray (or, more technically: if it has\nthe Py_TPFLAGS_SEQUENCE flag set in its tp_flags),\npush True onto the stack.  Otherwise, push False.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_SEQUENCE"
            };

        case "NOP":
            return {
                "html": "Do nothing code.  Used as a placeholder by the bytecode optimizer, and to\ngenerate line tracing events.",
                "tooltip": "Do nothing code.  Used as a placeholder by the bytecode optimizer, and to\ngenerate line tracing events.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-NOP"
            };

        case "POP_EXCEPT":
            return {
                "html": "Pops a value from the stack, which is used to restore the exception state.\nChanged in version 3.11: Exception representation on the stack now consist of one, not three, items.",
                "tooltip": "Pops a value from the stack, which is used to restore the exception state.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_EXCEPT"
            };

        case "POP_JUMP_BACKWARD_IF_FALSE":
            return {
                "html": "If TOS is false, decrements the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is false, decrements the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_FALSE"
            };

        case "POP_JUMP_BACKWARD_IF_NONE":
            return {
                "html": "If TOS is None, decrements the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is None, decrements the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_NONE"
            };

        case "POP_JUMP_BACKWARD_IF_NOT_NONE":
            return {
                "html": "If TOS is not None, decrements the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is not None, decrements the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_NOT_NONE"
            };

        case "POP_JUMP_BACKWARD_IF_TRUE":
            return {
                "html": "If TOS is true, decrements the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is true, decrements the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_BACKWARD_IF_TRUE"
            };

        case "POP_JUMP_FORWARD_IF_FALSE":
            return {
                "html": "If TOS is false, increments the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is false, increments the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_FORWARD_IF_FALSE"
            };

        case "POP_JUMP_FORWARD_IF_NONE":
            return {
                "html": "If TOS is None, increments the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is None, increments the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NONE"
            };

        case "POP_JUMP_FORWARD_IF_NOT_NONE":
            return {
                "html": "If TOS is not None, increments the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is not None, increments the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_FORWARD_IF_NOT_NONE"
            };

        case "POP_JUMP_FORWARD_IF_TRUE":
            return {
                "html": "If TOS is true, increments the bytecode counter by delta.  TOS is popped.\nNew in version 3.11.",
                "tooltip": "If TOS is true, increments the bytecode counter by delta.  TOS is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE"
            };

        case "POP_TOP":
            return {
                "html": "Removes the top-of-stack (TOS) item.",
                "tooltip": "Removes the top-of-stack (TOS) item.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_TOP"
            };

        case "PRECALL":
            return {
                "html": "Prefixes CALL. Logically this is a no op.\nIt exists to enable effective specialization of calls.\nargc is the number of arguments as described in CALL.\nNew in version 3.11.",
                "tooltip": "Prefixes CALL. Logically this is a no op.\nIt exists to enable effective specialization of calls.\nargc is the number of arguments as described in CALL.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PRECALL"
            };

        case "PREP_RERAISE_STAR":
            return {
                "html": "Combines the raised and reraised exceptions list from TOS, into an exception\ngroup to propagate from a try-except* block. Uses the original exception\ngroup from TOS1 to reconstruct the structure of reraised exceptions. Pops\ntwo items from the stack and pushes the exception to reraise or None\nif there isn\u2019t one.\nNew in version 3.11.",
                "tooltip": "Combines the raised and reraised exceptions list from TOS, into an exception\ngroup to propagate from a try-except* block. Uses the original exception\ngroup from TOS1 to reconstruct the structure of reraised exceptions. Pops\ntwo items from the stack and pushes the exception to reraise or None\nif there isn\u2019t one.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PREP_RERAISE_STAR"
            };

        case "PRINT_EXPR":
            return {
                "html": "Implements the expression statement for the interactive mode.  TOS is removed\nfrom the stack and printed.  In non-interactive mode, an expression statement\nis terminated with POP_TOP.",
                "tooltip": "Implements the expression statement for the interactive mode.  TOS is removed\nfrom the stack and printed.  In non-interactive mode, an expression statement\nis terminated with POP_TOP.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PRINT_EXPR"
            };

        case "PUSH_EXC_INFO":
            return {
                "html": "Pops a value from the stack. Pushes the current exception to the top of the stack.\nPushes the value originally popped back to the stack.\nUsed in exception handlers.\nNew in version 3.11.",
                "tooltip": "Pops a value from the stack. Pushes the current exception to the top of the stack.\nPushes the value originally popped back to the stack.\nUsed in exception handlers.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PUSH_EXC_INFO"
            };

        case "PUSH_NULL":
            return {
                "html": "Pushes a NULL to the stack.\nUsed in the call sequence to match the NULL pushed by\nLOAD_METHOD for non-method calls.\nNew in version 3.11.",
                "tooltip": "Pushes a NULL to the stack.\nUsed in the call sequence to match the NULL pushed by\nLOAD_METHOD for non-method calls.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PUSH_NULL"
            };

        case "RAISE_VARARGS":
            return {
                "html": "Raises an exception using one of the 3 forms of the raise statement,\ndepending on the value of argc:\n0: raise (re-raise previous exception)\n1: raise TOS (raise exception instance or type at TOS)\n2: raise TOS1 from TOS (raise exception instance or type at TOS1\nwith __cause__ set to TOS)",
                "tooltip": "Raises an exception using one of the 3 forms of the raise statement,\ndepending on the value of argc",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RAISE_VARARGS"
            };

        case "RERAISE":
            return {
                "html": "Re-raises the exception currently on top of the stack. If oparg is non-zero,\npops an additional value from the stack which is used to set f_lasti\nof the current frame.\nNew in version 3.9.\nChanged in version 3.11: Exception representation on the stack now consist of one, not three, items.",
                "tooltip": "Re-raises the exception currently on top of the stack. If oparg is non-zero,\npops an additional value from the stack which is used to set f_lasti\nof the current frame.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RERAISE"
            };

        case "RESUME":
            return {
                "html": "A no-op. Performs internal tracing, debugging and optimization checks.\nThe where operand marks where the RESUME occurs:\n0 The start of a function\n1 After a yield expression\n2 After a yield from expression\n3 After an await expression\nNew in version 3.11.",
                "tooltip": "A no-op. Performs internal tracing, debugging and optimization checks.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RESUME"
            };

        case "RETURN_GENERATOR":
            return {
                "html": "Create a generator, coroutine, or async generator from the current frame.\nClear the current frame and return the newly created generator.\nNew in version 3.11.",
                "tooltip": "Create a generator, coroutine, or async generator from the current frame.\nClear the current frame and return the newly created generator.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RETURN_GENERATOR"
            };

        case "RETURN_VALUE":
            return {
                "html": "Returns with TOS to the caller of the function.",
                "tooltip": "Returns with TOS to the caller of the function.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RETURN_VALUE"
            };

        case "SEND":
            return {
                "html": "Sends None to the sub-generator of this generator.\nUsed in yield from and await statements.\nNew in version 3.11.",
                "tooltip": "Sends None to the sub-generator of this generator.\nUsed in yield from and await statements.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SEND"
            };

        case "SETUP_ANNOTATIONS":
            return {
                "html": "Checks whether __annotations__ is defined in locals(), if not it is\nset up to an empty dict. This opcode is only emitted if a class\nor module body contains variable annotations\nstatically.\nNew in version 3.6.",
                "tooltip": "Checks whether __annotations__ is defined in locals(), if not it is\nset up to an empty dict. This opcode is only emitted if a class\nor module body contains variable annotations\nstatically.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SETUP_ANNOTATIONS"
            };

        case "SET_ADD":
            return {
                "html": "Calls set.add(TOS1[-i], TOS).  Used to implement set comprehensions.",
                "tooltip": "Calls set.add(TOS1[-i], TOS).  Used to implement set comprehensions.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SET_ADD"
            };

        case "SET_UPDATE":
            return {
                "html": "Calls set.update(TOS1[-i], TOS).  Used to build sets.\nNew in version 3.9.",
                "tooltip": "Calls set.update(TOS1[-i], TOS).  Used to build sets.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SET_UPDATE"
            };

        case "STORE_ATTR":
            return {
                "html": "Implements TOS.name = TOS1, where namei is the index of name in\nco_names.",
                "tooltip": "Implements TOS.name = TOS1, where namei is the index of name in\nco_names.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_ATTR"
            };

        case "STORE_DEREF":
            return {
                "html": "Stores TOS into the cell contained in slot i of the \u201cfast locals\u201d\nstorage.\nChanged in version 3.11: i is no longer offset by the length of co_varnames.",
                "tooltip": "Stores TOS into the cell contained in slot i of the \u201cfast locals\u201d\nstorage.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_DEREF"
            };

        case "STORE_FAST":
            return {
                "html": "Stores TOS into the local co_varnames[var_num].",
                "tooltip": "Stores TOS into the local co_varnames[var_num].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_FAST"
            };

        case "STORE_GLOBAL":
            return {
                "html": "Works as STORE_NAME, but stores the name as a global.",
                "tooltip": "Works as STORE_NAME, but stores the name as a global.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_GLOBAL"
            };

        case "STORE_NAME":
            return {
                "html": "Implements name = TOS. namei is the index of name in the attribute\nco_names of the code object. The compiler tries to use\nSTORE_FAST or STORE_GLOBAL if possible.",
                "tooltip": "Implements name = TOS. namei is the index of name in the attribute\nco_names of the code object. The compiler tries to use\nSTORE_FAST or STORE_GLOBAL if possible.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_NAME"
            };

        case "STORE_SUBSCR":
            return {
                "html": "Implements TOS1[TOS] = TOS2.",
                "tooltip": "Implements TOS1[TOS] = TOS2.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_SUBSCR"
            };

        case "SWAP":
            return {
                "html": "Swap TOS with the item at position i.\nNew in version 3.11.",
                "tooltip": "Swap TOS with the item at position i.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SWAP"
            };

        case "UNARY_INVERT":
            return {
                "html": "Implements TOS = ~TOS.",
                "tooltip": "Implements TOS = ~TOS.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_INVERT"
            };

        case "UNARY_NEGATIVE":
            return {
                "html": "Implements TOS = -TOS.",
                "tooltip": "Implements TOS = -TOS.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_NEGATIVE"
            };

        case "UNARY_NOT":
            return {
                "html": "Implements TOS = not TOS.",
                "tooltip": "Implements TOS = not TOS.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_NOT"
            };

        case "UNARY_POSITIVE":
            return {
                "html": "Implements TOS = +TOS.",
                "tooltip": "Implements TOS = +TOS.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_POSITIVE"
            };

        case "UNPACK_EX":
            return {
                "html": "Implements assignment with a starred target: Unpacks an iterable in TOS into\nindividual values, where the total number of values can be smaller than the\nnumber of items in the iterable: one of the new values will be a list of all\nleftover items.\nThe low byte of counts is the number of values before the list value, the\nhigh byte of counts the number of values after it.  The resulting values\nare put onto the stack right-to-left.",
                "tooltip": "Implements assignment with a starred target: Unpacks an iterable in TOS into\nindividual values, where the total number of values can be smaller than the\nnumber of items in the iterable: one of the new values will be a list of all\nleftover items.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNPACK_EX"
            };

        case "UNPACK_SEQUENCE":
            return {
                "html": "Unpacks TOS into count individual values, which are put onto the stack\nright-to-left.",
                "tooltip": "Unpacks TOS into count individual values, which are put onto the stack\nright-to-left.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNPACK_SEQUENCE"
            };

        case "WITH_EXCEPT_START":
            return {
                "html": "Calls the function in position 4 on the stack with arguments (type, val, tb)\nrepresenting the exception at the top of the stack.\nUsed to implement the call context_manager.__exit__(*exc_info()) when an exception\nhas occurred in a with statement.\nNew in version 3.9.\nChanged in version 3.11: The __exit__ function is in position 4 of the stack rather than 7.\nException representation on the stack now consist of one, not three, items.",
                "tooltip": "Calls the function in position 4 on the stack with arguments (type, val, tb)\nrepresenting the exception at the top of the stack.\nUsed to implement the call context_manager.__exit__(*exc_info()) when an exception\nhas occurred in a with statement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-WITH_EXCEPT_START"
            };

        case "YIELD_VALUE":
            return {
                "html": "Pops TOS and yields it from a generator.",
                "tooltip": "Pops TOS and yields it from a generator.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-YIELD_VALUE"
            };


    }
}
