import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "BEFORE_ASYNC_WITH":
            return {
                "html": "<p>Resolves <code>__aenter__</code> and <code>__aexit__</code> from <code>STACK[-1]</code>.\nPushes <code>__aexit__</code> and result of <code>__aenter__()</code> to the stack:</p>\n<pre>STACK.extend((__aexit__, __aenter__())\n</pre>\n\n\n<p>Added in version 3.5.</p>\n\n",
                "tooltip": "Resolves __aenter__ and __aexit__ from STACK[-1].\nPushes __aexit__ and result of __aenter__() to the stack",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BEFORE_ASYNC_WITH"
            };

        case "BEFORE_WITH":
            return {
                "html": "<p>This opcode performs several operations before a with block starts.  First,\nit loads <code>__exit__()</code> from the context manager and pushes it onto\nthe stack for later use by <code>WITH_EXCEPT_START</code>.  Then,\n<code>__enter__()</code> is called. Finally, the result of calling the\n<code>__enter__()</code> method is pushed onto the stack.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "This opcode performs several operations before a with block starts.  First,\nit loads __exit__() from the context manager and pushes it onto\nthe stack for later use by WITH_EXCEPT_START.  Then,\n__enter__() is called. Finally, the result of calling the\n__enter__() method is pushed onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BEFORE_WITH"
            };

        case "BINARY_OP":
            return {
                "html": "<p>Implements the binary and in-place operators (depending on the value of\nop):</p>\n<pre>rhs = STACK.pop()\nlhs = STACK.pop()\nSTACK.append(lhs op rhs)\n</pre>\n\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Implements the binary and in-place operators (depending on the value of\nop)",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BINARY_OP"
            };

        case "BINARY_SLICE":
            return {
                "html": "<p>Implements:</p>\n<pre>end = STACK.pop()\nstart = STACK.pop()\ncontainer = STACK.pop()\nSTACK.append(container[start:end])\n</pre>\n\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BINARY_SLICE"
            };

        case "BINARY_SUBSCR":
            return {
                "html": "<p>Implements:</p>\n<pre>key = STACK.pop()\ncontainer = STACK.pop()\nSTACK.append(container[key])\n</pre>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BINARY_SUBSCR"
            };

        case "BUILD_CONST_KEY_MAP":
            return {
                "html": "<p>The version of <code>BUILD_MAP</code> specialized for constant keys. Pops the\ntop element on the stack which contains a tuple of keys, then starting from\n<code>STACK[-2]</code>, pops count values to form values in the built dictionary.</p>\n\n<p>Added in version 3.6.</p>\n\n",
                "tooltip": "The version of BUILD_MAP specialized for constant keys. Pops the\ntop element on the stack which contains a tuple of keys, then starting from\nSTACK[-2], pops count values to form values in the built dictionary.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_CONST_KEY_MAP"
            };

        case "BUILD_LIST":
            return {
                "html": "<p>Works as <code>BUILD_TUPLE</code>, but creates a list.</p>\n",
                "tooltip": "Works as BUILD_TUPLE, but creates a list.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_LIST"
            };

        case "BUILD_MAP":
            return {
                "html": "<p>Pushes a new dictionary object onto the stack.  Pops <code>2 * count</code> items\nso that the dictionary holds count entries:\n<code>{..., STACK[-4]: STACK[-3], STACK[-2]: STACK[-1]}</code>.</p>\n\n<p>Changed in version 3.5: The dictionary is created from stack items instead of creating an\nempty dictionary pre-sized to hold count items.</p>\n\n",
                "tooltip": "Pushes a new dictionary object onto the stack.  Pops 2 * count items\nso that the dictionary holds count entries:\n{..., STACK[-4]: STACK[-3], STACK[-2]: STACK[-1]}.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_MAP"
            };

        case "BUILD_SET":
            return {
                "html": "<p>Works as <code>BUILD_TUPLE</code>, but creates a set.</p>\n",
                "tooltip": "Works as BUILD_TUPLE, but creates a set.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_SET"
            };

        case "BUILD_SLICE":
            return {
                "html": "<p>Pushes a slice object on the stack.  argc must be 2 or 3.  If it is 2, implements:</p>\n<pre>end = STACK.pop()\nstart = STACK.pop()\nSTACK.append(slice(start, end))\n</pre>\n\n<p>if it is 3, implements:</p>\n<pre>step = STACK.pop()\nend = STACK.pop()\nstart = STACK.pop()\nSTACK.append(slice(start, end, step))\n</pre>\n\n<p>See the <code>slice()</code> built-in function for more information.</p>\n",
                "tooltip": "Pushes a slice object on the stack.  argc must be 2 or 3.  If it is 2, implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_SLICE"
            };

        case "BUILD_STRING":
            return {
                "html": "<p>Concatenates count strings from the stack and pushes the resulting string\nonto the stack.</p>\n\n<p>Added in version 3.6.</p>\n\n",
                "tooltip": "Concatenates count strings from the stack and pushes the resulting string\nonto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING"
            };

        case "BUILD_TUPLE":
            return {
                "html": "<p>Creates a tuple consuming count items from the stack, and pushes the\nresulting tuple onto the stack:</p>\n<pre>if count == 0:\n    value = ()\nelse:\n    value = tuple(STACK[-count:])\n    STACK = STACK[:-count]\n\nSTACK.append(value)\n</pre>\n\n",
                "tooltip": "Creates a tuple consuming count items from the stack, and pushes the\nresulting tuple onto the stack",
                "url": "https://docs.python.org/3/library/dis.html#opcode-BUILD_TUPLE"
            };

        case "CACHE":
            return {
                "html": "<p>Rather than being an actual instruction, this opcode is used to mark extra\nspace for the interpreter to cache useful data directly in the bytecode\nitself. It is automatically hidden by all <code>dis</code> utilities, but can be\nviewed with <code>show_caches=True</code>.</p>\n<p>Logically, this space is part of the preceding instruction. Many opcodes\nexpect to be followed by an exact number of caches, and will instruct the\ninterpreter to skip over them at runtime.</p>\n<p>Populated caches can look like arbitrary instructions, so great care should\nbe taken when reading or modifying raw, adaptive bytecode containing\nquickened data.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Rather than being an actual instruction, this opcode is used to mark extra\nspace for the interpreter to cache useful data directly in the bytecode\nitself. It is automatically hidden by all dis utilities, but can be\nviewed with show_caches=True.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CACHE"
            };

        case "CALL":
            return {
                "html": "<p>Calls a callable object with the number of arguments specified by <code>argc</code>,\nincluding the named arguments specified by the preceding\n<code>KW_NAMES</code>, if any.\nOn the stack are (in ascending order), either:</p>\n<ul>\n<li><p>NULL</p></li>\n<li><p>The callable</p></li>\n<li><p>The positional arguments</p></li>\n<li><p>The named arguments</p></li>\n</ul>\n<p>or:</p>\n<ul>\n<li><p>The callable</p></li>\n<li><p><code>self</code></p></li>\n<li><p>The remaining positional arguments</p></li>\n<li><p>The named arguments</p></li>\n</ul>\n<p><code>argc</code> is the total of the positional and named arguments, excluding\n<code>self</code> when a <code>NULL</code> is not present.</p>\n<p><code>CALL</code> pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Calls a callable object with the number of arguments specified by argc,\nincluding the named arguments specified by the preceding\nKW_NAMES, if any.\nOn the stack are (in ascending order), either",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL"
            };

        case "CALL_FUNCTION_EX":
            return {
                "html": "<p>Calls a callable object with variable set of positional and keyword\narguments.  If the lowest bit of flags is set, the top of the stack\ncontains a mapping object containing additional keyword arguments.\nBefore the callable is called, the mapping object and iterable object\nare each \u201cunpacked\u201d and their contents passed in as keyword and\npositional arguments respectively.\n<code>CALL_FUNCTION_EX</code> pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.</p>\n\n<p>Added in version 3.6.</p>\n\n",
                "tooltip": "Calls a callable object with variable set of positional and keyword\narguments.  If the lowest bit of flags is set, the top of the stack\ncontains a mapping object containing additional keyword arguments.\nBefore the callable is called, the mapping object and iterable object\nare each \u201cunpacked\u201d and their contents passed in as keyword and\npositional arguments respectively.\nCALL_FUNCTION_EX pops all arguments and the callable object off the stack,\ncalls the callable object with those arguments, and pushes the return value\nreturned by the callable object.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL_FUNCTION_EX"
            };

        case "CALL_INTRINSIC_1":
            return {
                "html": "<p>Calls an intrinsic function with one argument. Passes <code>STACK[-1]</code> as the\nargument and sets <code>STACK[-1]</code> to the result. Used to implement\nfunctionality that is not performance critical.</p>\n<p>The operand determines which intrinsic function is called:</p>\n<table>\n\n<tr><th><p>Operand</p></th>\n<th><p>Description</p></th>\n</tr>\n\n\n<tr><td><p><code>INTRINSIC_1_INVALID</code></p></td>\n<td><p>Not valid</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_PRINT</code></p></td>\n<td><p>Prints the argument to standard\nout. Used in the REPL.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_IMPORT_STAR</code></p></td>\n<td><p>Performs <code>import *</code> for the\nnamed module.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_STOPITERATION_ERROR</code></p></td>\n<td><p>Extracts the return value from a\n<code>StopIteration</code> exception.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_ASYNC_GEN_WRAP</code></p></td>\n<td><p>Wraps an async generator value</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_UNARY_POSITIVE</code></p></td>\n<td><p>Performs the unary <code>+</code>\noperation</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_LIST_TO_TUPLE</code></p></td>\n<td><p>Converts a list to a tuple</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_TYPEVAR</code></p></td>\n<td><p>Creates a <code>typing.TypeVar</code></p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_PARAMSPEC</code></p></td>\n<td><p>Creates a\n<code>typing.ParamSpec</code></p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_TYPEVARTUPLE</code></p></td>\n<td><p>Creates a\n<code>typing.TypeVarTuple</code></p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_SUBSCRIPT_GENERIC</code></p></td>\n<td><p>Returns <code>typing.Generic</code>\nsubscripted with the argument</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_TYPEALIAS</code></p></td>\n<td><p>Creates a\n<code>typing.TypeAliasType</code>;\nused in the <code>type</code>\nstatement. The argument is a tuple\nof the type alias\u2019s name,\ntype parameters, and value.</p></td>\n</tr>\n\n</table>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Calls an intrinsic function with one argument. Passes STACK[-1] as the\nargument and sets STACK[-1] to the result. Used to implement\nfunctionality that is not performance critical.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL_INTRINSIC_1"
            };

        case "CALL_INTRINSIC_2":
            return {
                "html": "<p>Calls an intrinsic function with two arguments. Used to implement functionality\nthat is not performance critical:</p>\n<pre>arg2 = STACK.pop()\narg1 = STACK.pop()\nresult = intrinsic2(arg1, arg2)\nSTACK.push(result)\n</pre>\n\n<p>The operand determines which intrinsic function is called:</p>\n<table>\n\n<tr><th><p>Operand</p></th>\n<th><p>Description</p></th>\n</tr>\n\n\n<tr><td><p><code>INTRINSIC_2_INVALID</code></p></td>\n<td><p>Not valid</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_PREP_RERAISE_STAR</code></p></td>\n<td><p>Calculates the\n<code>ExceptionGroup</code> to raise\nfrom a <code>try-except*</code>.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_TYPEVAR_WITH_BOUND</code></p></td>\n<td><p>Creates a <code>typing.TypeVar</code>\nwith a bound.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_TYPEVAR_WITH_CONSTRAINTS</code></p></td>\n<td><p>Creates a\n<code>typing.TypeVar</code> with\nconstraints.</p></td>\n</tr>\n<tr><td><p><code>INTRINSIC_SET_FUNCTION_TYPE_PARAMS</code></p></td>\n<td><p>Sets the <code>__type_params__</code>\nattribute of a function.</p></td>\n</tr>\n\n</table>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Calls an intrinsic function with two arguments. Used to implement functionality\nthat is not performance critical",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CALL_INTRINSIC_2"
            };

        case "CHECK_EG_MATCH":
            return {
                "html": "<p>Performs exception matching for <code>except*</code>. Applies <code>split(STACK[-1])</code> on\nthe exception group representing <code>STACK[-2]</code>.</p>\n<p>In case of a match, pops two items from the stack and pushes the\nnon-matching subgroup (<code>None</code> in case of full match) followed by the\nmatching subgroup. When there is no match, pops one item (the match\ntype) and pushes <code>None</code>.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Performs exception matching for except*. Applies split(STACK[-1]) on\nthe exception group representing STACK[-2].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CHECK_EG_MATCH"
            };

        case "CHECK_EXC_MATCH":
            return {
                "html": "<p>Performs exception matching for <code>except</code>. Tests whether the <code>STACK[-2]</code>\nis an exception matching <code>STACK[-1]</code>. Pops <code>STACK[-1]</code> and pushes the boolean\nresult of the test.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Performs exception matching for except. Tests whether the STACK[-2]\nis an exception matching STACK[-1]. Pops STACK[-1] and pushes the boolean\nresult of the test.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CHECK_EXC_MATCH"
            };

        case "CLEANUP_THROW":
            return {
                "html": "<p>Handles an exception raised during a <code>throw()</code> or\n<code>close()</code> call through the current frame.  If <code>STACK[-1]</code> is an\ninstance of <code>StopIteration</code>, pop three values from the stack and push\nits <code>value</code> member.  Otherwise, re-raise <code>STACK[-1]</code>.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Handles an exception raised during a throw() or\nclose() call through the current frame.  If STACK[-1] is an\ninstance of StopIteration, pop three values from the stack and push\nits value member.  Otherwise, re-raise STACK[-1].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CLEANUP_THROW"
            };

        case "COMPARE_OP":
            return {
                "html": "<p>Performs a Boolean operation.  The operation name can be found in\n<code>cmp_op[opname &gt;&gt; 4]</code>.</p>\n\n<p>Changed in version 3.12: The cmp_op index is now stored in the four-highest bits of oparg instead of the four-lowest bits of oparg.</p>\n\n",
                "tooltip": "Performs a Boolean operation.  The operation name can be found in\ncmp_op[opname >> 4].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COMPARE_OP"
            };

        case "CONTAINS_OP":
            return {
                "html": "<p>Performs <code>in</code> comparison, or <code>not in</code> if <code>invert</code> is 1.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Performs in comparison, or not in if invert is 1.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-CONTAINS_OP"
            };

        case "COPY":
            return {
                "html": "<p>Push the i-th item to the top of the stack without removing it from its original\nlocation:</p>\n<pre>assert i &gt; 0\nSTACK.append(STACK[-i])\n</pre>\n\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Push the i-th item to the top of the stack without removing it from its original\nlocation",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COPY"
            };

        case "COPY_FREE_VARS":
            return {
                "html": "<p>Copies the <code>n</code> free variables from the closure into the frame.\nRemoves the need for special code on the caller\u2019s side when calling\nclosures.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Copies the n free variables from the closure into the frame.\nRemoves the need for special code on the caller\u2019s side when calling\nclosures.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-COPY_FREE_VARS"
            };

        case "DELETE_ATTR":
            return {
                "html": "<p>Implements:</p>\n<pre>obj = STACK.pop()\ndel obj.name\n</pre>\n\n<p>where namei is the index of name into <code>co_names</code> of the\ncode object.</p>\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_ATTR"
            };

        case "DELETE_DEREF":
            return {
                "html": "<p>Empties the cell contained in slot <code>i</code> of the \u201cfast locals\u201d storage.\nUsed by the <code>del</code> statement.</p>\n\n<p>Added in version 3.2.</p>\n\n\n<p>Changed in version 3.11: <code>i</code> is no longer offset by the length of <code>co_varnames</code>.</p>\n\n",
                "tooltip": "Empties the cell contained in slot i of the \u201cfast locals\u201d storage.\nUsed by the del statement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_DEREF"
            };

        case "DELETE_FAST":
            return {
                "html": "<p>Deletes local <code>co_varnames[var_num]</code>.</p>\n",
                "tooltip": "Deletes local co_varnames[var_num].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_FAST"
            };

        case "DELETE_GLOBAL":
            return {
                "html": "<p>Works as <code>DELETE_NAME</code>, but deletes a global name.</p>\n",
                "tooltip": "Works as DELETE_NAME, but deletes a global name.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_GLOBAL"
            };

        case "DELETE_NAME":
            return {
                "html": "<p>Implements <code>del name</code>, where namei is the index into <code>co_names</code>\nattribute of the code object.</p>\n",
                "tooltip": "Implements del name, where namei is the index into co_names\nattribute of the code object.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_NAME"
            };

        case "DELETE_SUBSCR":
            return {
                "html": "<p>Implements:</p>\n<pre>key = STACK.pop()\ncontainer = STACK.pop()\ndel container[key]\n</pre>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DELETE_SUBSCR"
            };

        case "DICT_MERGE":
            return {
                "html": "<p>Like <code>DICT_UPDATE</code> but raises an exception for duplicate keys.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Like DICT_UPDATE but raises an exception for duplicate keys.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DICT_MERGE"
            };

        case "DICT_UPDATE":
            return {
                "html": "<p>Implements:</p>\n<pre>map = STACK.pop()\ndict.update(STACK[-i], map)\n</pre>\n\n<p>Used to build dicts.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-DICT_UPDATE"
            };

        case "END_ASYNC_FOR":
            return {
                "html": "<p>Terminates an <code>async for</code> loop.  Handles an exception raised\nwhen awaiting a next item. The stack contains the async iterable in\n<code>STACK[-2]</code> and the raised exception in <code>STACK[-1]</code>. Both are popped.\nIf the exception is not <code>StopAsyncIteration</code>, it is re-raised.</p>\n\n<p>Added in version 3.8.</p>\n\n\n<p>Changed in version 3.11: Exception representation on the stack now consist of one, not three, items.</p>\n\n",
                "tooltip": "Terminates an async for loop.  Handles an exception raised\nwhen awaiting a next item. The stack contains the async iterable in\nSTACK[-2] and the raised exception in STACK[-1]. Both are popped.\nIf the exception is not StopAsyncIteration, it is re-raised.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-END_ASYNC_FOR"
            };

        case "END_FOR":
            return {
                "html": "<p>Removes the top two values from the stack.\nEquivalent to <code>POP_TOP</code>; <code>POP_TOP</code>.\nUsed to clean up at the end of loops, hence the name.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Removes the top two values from the stack.\nEquivalent to POP_TOP; POP_TOP.\nUsed to clean up at the end of loops, hence the name.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-END_FOR"
            };

        case "END_SEND":
            return {
                "html": "<p>Implements <code>del STACK[-2]</code>.\nUsed to clean up when a generator exits.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Implements del STACK[-2].\nUsed to clean up when a generator exits.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-END_SEND"
            };

        case "EXTENDED_ARG":
            return {
                "html": "<p>Prefixes any opcode which has an argument too big to fit into the default one\nbyte. ext holds an additional byte which act as higher bits in the argument.\nFor each opcode, at most three prefixal <code>EXTENDED_ARG</code> are allowed, forming\nan argument from two-byte to four-byte.</p>\n",
                "tooltip": "Prefixes any opcode which has an argument too big to fit into the default one\nbyte. ext holds an additional byte which act as higher bits in the argument.\nFor each opcode, at most three prefixal EXTENDED_ARG are allowed, forming\nan argument from two-byte to four-byte.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-EXTENDED_ARG"
            };

        case "FORMAT_VALUE":
            return {
                "html": "<p>Used for implementing formatted literal strings (f-strings).  Pops\nan optional fmt_spec from the stack, then a required value.\nflags is interpreted as follows:</p>\n<ul>\n<li><p><code>(flags &amp; 0x03) == 0x00</code>: value is formatted as-is.</p></li>\n<li><p><code>(flags &amp; 0x03) == 0x01</code>: call <code>str()</code> on value before\nformatting it.</p></li>\n<li><p><code>(flags &amp; 0x03) == 0x02</code>: call <code>repr()</code> on value before\nformatting it.</p></li>\n<li><p><code>(flags &amp; 0x03) == 0x03</code>: call <code>ascii()</code> on value before\nformatting it.</p></li>\n<li><p><code>(flags &amp; 0x04) == 0x04</code>: pop fmt_spec from the stack and use\nit, else use an empty fmt_spec.</p></li>\n</ul>\n<p>Formatting is performed using <code>PyObject_Format()</code>.  The\nresult is pushed on the stack.</p>\n\n<p>Added in version 3.6.</p>\n\n",
                "tooltip": "Used for implementing formatted literal strings (f-strings).  Pops\nan optional fmt_spec from the stack, then a required value.\nflags is interpreted as follows",
                "url": "https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE"
            };

        case "FOR_ITER":
            return {
                "html": "<p><code>STACK[-1]</code> is an iterator.  Call its <code>__next__()</code> method.\nIf this yields a new value, push it on the stack (leaving the iterator below\nit).  If the iterator indicates it is exhausted then the byte code counter is\nincremented by delta.</p>\n\n<p>Changed in version 3.12: Up until 3.11 the iterator was popped when it was exhausted.</p>\n\n",
                "tooltip": "STACK[-1] is an iterator.  Call its __next__() method.\nIf this yields a new value, push it on the stack (leaving the iterator below\nit).  If the iterator indicates it is exhausted then the byte code counter is\nincremented by delta.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-FOR_ITER"
            };

        case "GET_AITER":
            return {
                "html": "<p>Implements <code>STACK[-1] = STACK[-1].__aiter__()</code>.</p>\n\n<p>Added in version 3.5.</p>\n\n\n<p>Changed in version 3.7: Returning awaitable objects from <code>__aiter__</code> is no longer\nsupported.</p>\n\n",
                "tooltip": "Implements STACK[-1] = STACK[-1].__aiter__().",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_AITER"
            };

        case "GET_ANEXT":
            return {
                "html": "<p>Implement <code>STACK.append(get_awaitable(STACK[-1].__anext__()))</code> to the stack.\nSee <code>GET_AWAITABLE</code> for details about <code>get_awaitable</code>.</p>\n\n<p>Added in version 3.5.</p>\n\n",
                "tooltip": "Implement STACK.append(get_awaitable(STACK[-1].__anext__())) to the stack.\nSee GET_AWAITABLE for details about get_awaitable.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_ANEXT"
            };

        case "GET_AWAITABLE":
            return {
                "html": "<p>Implements <code>STACK[-1] = get_awaitable(STACK[-1])</code>, where <code>get_awaitable(o)</code>\nreturns <code>o</code> if <code>o</code> is a coroutine object or a generator object with\nthe <code>CO_ITERABLE_COROUTINE</code> flag, or resolves\n<code>o.__await__</code>.</p>\n\n<p>If the <code>where</code> operand is nonzero, it indicates where the instruction\noccurs:</p>\n<ul>\n<li><p><code>1</code>: After a call to <code>__aenter__</code></p></li>\n<li><p><code>2</code>: After a call to <code>__aexit__</code></p></li>\n</ul>\n\n\n<p>Added in version 3.5.</p>\n\n\n<p>Changed in version 3.11: Previously, this instruction did not have an oparg.</p>\n\n",
                "tooltip": "Implements STACK[-1] = get_awaitable(STACK[-1]), where get_awaitable(o)\nreturns o if o is a coroutine object or a generator object with\nthe CO_ITERABLE_COROUTINE flag, or resolves\no.__await__.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_AWAITABLE"
            };

        case "GET_ITER":
            return {
                "html": "<p>Implements <code>STACK[-1] = iter(STACK[-1])</code>.</p>\n",
                "tooltip": "Implements STACK[-1] = iter(STACK[-1]).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_ITER"
            };

        case "GET_LEN":
            return {
                "html": "<p>Perform <code>STACK.append(len(STACK[-1]))</code>.</p>\n\n<p>Added in version 3.10.</p>\n\n",
                "tooltip": "Perform STACK.append(len(STACK[-1])).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_LEN"
            };

        case "GET_YIELD_FROM_ITER":
            return {
                "html": "<p>If <code>STACK[-1]</code> is a generator iterator or coroutine object\nit is left as is.  Otherwise, implements <code>STACK[-1] = iter(STACK[-1])</code>.</p>\n\n<p>Added in version 3.5.</p>\n\n",
                "tooltip": "If STACK[-1] is a generator iterator or coroutine object\nit is left as is.  Otherwise, implements STACK[-1] = iter(STACK[-1]).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-GET_YIELD_FROM_ITER"
            };

        case "HAVE_ARGUMENT":
            return {
                "html": "<p>This is not really an opcode.  It identifies the dividing line between\nopcodes in the range [0,255] which don\u2019t use their argument and those\nthat do (<code>&lt; HAVE_ARGUMENT</code> and <code>&gt;= HAVE_ARGUMENT</code>, respectively).</p>\n<p>If your application uses pseudo instructions, use the <code>hasarg</code>\ncollection instead.</p>\n\n<p>Changed in version 3.6: Now every instruction has an argument, but opcodes <code>&lt; HAVE_ARGUMENT</code>\nignore it. Before, only opcodes <code>&gt;= HAVE_ARGUMENT</code> had an argument.</p>\n\n\n<p>Changed in version 3.12: Pseudo instructions were added to the <code>dis</code> module, and for them\nit is not true that comparison with <code>HAVE_ARGUMENT</code> indicates whether\nthey use their arg.</p>\n\n",
                "tooltip": "This is not really an opcode.  It identifies the dividing line between\nopcodes in the range [0,255] which don\u2019t use their argument and those\nthat do (< HAVE_ARGUMENT and >= HAVE_ARGUMENT, respectively).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-HAVE_ARGUMENT"
            };

        case "IMPORT_FROM":
            return {
                "html": "<p>Loads the attribute <code>co_names[namei]</code> from the module found in <code>STACK[-1]</code>.\nThe resulting object is pushed onto the stack, to be subsequently stored by a\n<code>STORE_FAST</code> instruction.</p>\n",
                "tooltip": "Loads the attribute co_names[namei] from the module found in STACK[-1].\nThe resulting object is pushed onto the stack, to be subsequently stored by a\nSTORE_FAST instruction.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IMPORT_FROM"
            };

        case "IMPORT_NAME":
            return {
                "html": "<p>Imports the module <code>co_names[namei]</code>.  <code>STACK[-1]</code> and <code>STACK[-2]</code> are\npopped and provide the fromlist and level arguments of <code>__import__()</code>.\nThe module object is pushed onto the stack.  The current namespace is not affected: for a proper import statement, a subsequent <code>STORE_FAST</code> instruction\nmodifies the namespace.</p>\n",
                "tooltip": "Imports the module co_names[namei].  STACK[-1] and STACK[-2] are\npopped and provide the fromlist and level arguments of __import__().\nThe module object is pushed onto the stack.  The current namespace is not affected: for a proper import statement, a subsequent STORE_FAST instruction\nmodifies the namespace.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IMPORT_NAME"
            };

        case "IS_OP":
            return {
                "html": "<p>Performs <code>is</code> comparison, or <code>is not</code> if <code>invert</code> is 1.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Performs is comparison, or is not if invert is 1.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-IS_OP"
            };

        case "JUMP":
            return {
                "html": "",
                "tooltip": "",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP"
            };

        case "JUMP_BACKWARD":
            return {
                "html": "<p>Decrements bytecode counter by delta. Checks for interrupts.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Decrements bytecode counter by delta. Checks for interrupts.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_BACKWARD"
            };

        case "JUMP_BACKWARD_NO_INTERRUPT":
            return {
                "html": "<p>Decrements bytecode counter by delta. Does not check for interrupts.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Decrements bytecode counter by delta. Does not check for interrupts.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_BACKWARD_NO_INTERRUPT"
            };

        case "JUMP_FORWARD":
            return {
                "html": "<p>Increments bytecode counter by delta.</p>\n",
                "tooltip": "Increments bytecode counter by delta.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_FORWARD"
            };

        case "JUMP_NO_INTERRUPT":
            return {
                "html": "<p>Undirected relative jump instructions which are replaced by their\ndirected (forward/backward) counterparts by the assembler.</p>\n",
                "tooltip": "Undirected relative jump instructions which are replaced by their\ndirected (forward/backward) counterparts by the assembler.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-JUMP_NO_INTERRUPT"
            };

        case "KW_NAMES":
            return {
                "html": "<p>Prefixes <code>CALL</code>.\nStores a reference to <code>co_consts[consti]</code> into an internal variable\nfor use by <code>CALL</code>. <code>co_consts[consti]</code> must be a tuple of strings.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Prefixes CALL.\nStores a reference to co_consts[consti] into an internal variable\nfor use by CALL. co_consts[consti] must be a tuple of strings.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-KW_NAMES"
            };

        case "LIST_APPEND":
            return {
                "html": "<p>Implements:</p>\n<pre>item = STACK.pop()\nlist.append(STACK[-i], item)\n</pre>\n\n<p>Used to implement list comprehensions.</p>\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LIST_APPEND"
            };

        case "LIST_EXTEND":
            return {
                "html": "<p>Implements:</p>\n<pre>seq = STACK.pop()\nlist.extend(STACK[-i], seq)\n</pre>\n\n<p>Used to build lists.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LIST_EXTEND"
            };

        case "LOAD_ASSERTION_ERROR":
            return {
                "html": "<p>Pushes <code>AssertionError</code> onto the stack.  Used by the <code>assert</code>\nstatement.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Pushes AssertionError onto the stack.  Used by the assert\nstatement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_ASSERTION_ERROR"
            };

        case "LOAD_ATTR":
            return {
                "html": "<p>If the low bit of <code>namei</code> is not set, this replaces <code>STACK[-1]</code> with\n<code>getattr(STACK[-1], co_names[namei&gt;&gt;1])</code>.</p>\n<p>If the low bit of <code>namei</code> is set, this will attempt to load a method named\n<code>co_names[namei&gt;&gt;1]</code> from the <code>STACK[-1]</code> object. <code>STACK[-1]</code> is popped.\nThis bytecode distinguishes two cases: if <code>STACK[-1]</code> has a method with the\ncorrect name, the bytecode pushes the unbound method and <code>STACK[-1]</code>.\n<code>STACK[-1]</code> will be used as the first argument (<code>self</code>) by <code>CALL</code>\nwhen calling the unbound method. Otherwise, <code>NULL</code> and the object returned by\nthe attribute lookup are pushed.</p>\n\n<p>Changed in version 3.12: If the low bit of <code>namei</code> is set, then a <code>NULL</code> or <code>self</code> is\npushed to the stack before the attribute or unbound method respectively.</p>\n\n",
                "tooltip": "If the low bit of namei is not set, this replaces STACK[-1] with\ngetattr(STACK[-1], co_names[namei>>1]).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_ATTR"
            };

        case "LOAD_BUILD_CLASS":
            return {
                "html": "<p>Pushes <code>builtins.__build_class__()</code> onto the stack.  It is later called\nto construct a class.</p>\n",
                "tooltip": "Pushes builtins.__build_class__() onto the stack.  It is later called\nto construct a class.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_BUILD_CLASS"
            };

        case "LOAD_CLOSURE":
            return {
                "html": "<p>Pushes a reference to the cell contained in slot <code>i</code> of the \u201cfast locals\u201d\nstorage.  The name of the variable is <code>co_fastlocalnames[i]</code>.</p>\n<p>Note that <code>LOAD_CLOSURE</code> is effectively an alias for <code>LOAD_FAST</code>.\nIt exists to keep bytecode a little more readable.</p>\n\n<p>Changed in version 3.11: <code>i</code> is no longer offset by the length of <code>co_varnames</code>.</p>\n\n",
                "tooltip": "Pushes a reference to the cell contained in slot i of the \u201cfast locals\u201d\nstorage.  The name of the variable is co_fastlocalnames[i].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_CLOSURE"
            };

        case "LOAD_CONST":
            return {
                "html": "<p>Pushes <code>co_consts[consti]</code> onto the stack.</p>\n",
                "tooltip": "Pushes co_consts[consti] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_CONST"
            };

        case "LOAD_DEREF":
            return {
                "html": "<p>Loads the cell contained in slot <code>i</code> of the \u201cfast locals\u201d storage.\nPushes a reference to the object the cell contains on the stack.</p>\n\n<p>Changed in version 3.11: <code>i</code> is no longer offset by the length of <code>co_varnames</code>.</p>\n\n",
                "tooltip": "Loads the cell contained in slot i of the \u201cfast locals\u201d storage.\nPushes a reference to the object the cell contains on the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_DEREF"
            };

        case "LOAD_FAST":
            return {
                "html": "<p>Pushes a reference to the local <code>co_varnames[var_num]</code> onto the stack.</p>\n\n<p>Changed in version 3.12: This opcode is now only used in situations where the local variable is\nguaranteed to be initialized. It cannot raise <code>UnboundLocalError</code>.</p>\n\n",
                "tooltip": "Pushes a reference to the local co_varnames[var_num] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST"
            };

        case "LOAD_FAST_AND_CLEAR":
            return {
                "html": "<p>Pushes a reference to the local <code>co_varnames[var_num]</code> onto the stack (or\npushes <code>NULL</code> onto the stack if the local variable has not been\ninitialized) and sets <code>co_varnames[var_num]</code> to <code>NULL</code>.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Pushes a reference to the local co_varnames[var_num] onto the stack (or\npushes NULL onto the stack if the local variable has not been\ninitialized) and sets co_varnames[var_num] to NULL.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST_AND_CLEAR"
            };

        case "LOAD_FAST_CHECK":
            return {
                "html": "<p>Pushes a reference to the local <code>co_varnames[var_num]</code> onto the stack,\nraising an <code>UnboundLocalError</code> if the local variable has not been\ninitialized.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Pushes a reference to the local co_varnames[var_num] onto the stack,\nraising an UnboundLocalError if the local variable has not been\ninitialized.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FAST_CHECK"
            };

        case "LOAD_FROM_DICT_OR_DEREF":
            return {
                "html": "<p>Pops a mapping off the stack and looks up the name associated with\nslot <code>i</code> of the \u201cfast locals\u201d storage in this mapping.\nIf the name is not found there, loads it from the cell contained in\nslot <code>i</code>, similar to <code>LOAD_DEREF</code>. This is used for loading\nfree variables in class bodies (which previously used\n<code>LOAD_CLASSDEREF</code>) and in\nannotation scopes within class bodies.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Pops a mapping off the stack and looks up the name associated with\nslot i of the \u201cfast locals\u201d storage in this mapping.\nIf the name is not found there, loads it from the cell contained in\nslot i, similar to LOAD_DEREF. This is used for loading\nfree variables in class bodies (which previously used\nLOAD_CLASSDEREF) and in\nannotation scopes within class bodies.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FROM_DICT_OR_DEREF"
            };

        case "LOAD_FROM_DICT_OR_GLOBALS":
            return {
                "html": "<p>Pops a mapping off the stack and looks up the value for <code>co_names[namei]</code>.\nIf the name is not found there, looks it up in the globals and then the builtins,\nsimilar to <code>LOAD_GLOBAL</code>.\nThis is used for loading global variables in\nannotation scopes within class bodies.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Pops a mapping off the stack and looks up the value for co_names[namei].\nIf the name is not found there, looks it up in the globals and then the builtins,\nsimilar to LOAD_GLOBAL.\nThis is used for loading global variables in\nannotation scopes within class bodies.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_FROM_DICT_OR_GLOBALS"
            };

        case "LOAD_GLOBAL":
            return {
                "html": "<p>Loads the global named <code>co_names[namei&gt;&gt;1]</code> onto the stack.</p>\n\n<p>Changed in version 3.11: If the low bit of <code>namei</code> is set, then a <code>NULL</code> is pushed to the\nstack before the global variable.</p>\n\n",
                "tooltip": "Loads the global named co_names[namei>>1] onto the stack.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_GLOBAL"
            };

        case "LOAD_LOCALS":
            return {
                "html": "<p>Pushes a reference to the locals dictionary onto the stack.  This is used\nto prepare namespace dictionaries for <code>LOAD_FROM_DICT_OR_DEREF</code>\nand <code>LOAD_FROM_DICT_OR_GLOBALS</code>.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Pushes a reference to the locals dictionary onto the stack.  This is used\nto prepare namespace dictionaries for LOAD_FROM_DICT_OR_DEREF\nand LOAD_FROM_DICT_OR_GLOBALS.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_LOCALS"
            };

        case "LOAD_METHOD":
            return {
                "html": "<p>Optimized unbound method lookup. Emitted as a <code>LOAD_ATTR</code> opcode\nwith a flag set in the arg.</p>\n",
                "tooltip": "Optimized unbound method lookup. Emitted as a LOAD_ATTR opcode\nwith a flag set in the arg.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_METHOD"
            };

        case "LOAD_NAME":
            return {
                "html": "<p>Pushes the value associated with <code>co_names[namei]</code> onto the stack.\nThe name is looked up within the locals, then the globals, then the builtins.</p>\n",
                "tooltip": "Pushes the value associated with co_names[namei] onto the stack.\nThe name is looked up within the locals, then the globals, then the builtins.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_NAME"
            };

        case "LOAD_SUPER_ATTR":
            return {
                "html": "<p>This opcode implements <code>super()</code>, both in its zero-argument and\ntwo-argument forms (e.g. <code>super().method()</code>, <code>super().attr</code> and\n<code>super(cls, self).method()</code>, <code>super(cls, self).attr</code>).</p>\n<p>It pops three values from the stack (from top of stack down):\n- <code>self</code>: the first argument to the current method\n-  <code>cls</code>: the class within which the current method was defined\n-  the global <code>super</code></p>\n<p>With respect to its argument, it works similarly to <code>LOAD_ATTR</code>,\nexcept that <code>namei</code> is shifted left by 2 bits instead of 1.</p>\n<p>The low bit of <code>namei</code> signals to attempt a method load, as with\n<code>LOAD_ATTR</code>, which results in pushing <code>NULL</code> and the loaded method.\nWhen it is unset a single value is pushed to the stack.</p>\n<p>The second-low bit of <code>namei</code>, if set, means that this was a two-argument\ncall to <code>super()</code> (unset means zero-argument).</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "This opcode implements super(), both in its zero-argument and\ntwo-argument forms (e.g. super().method(), super().attr and\nsuper(cls, self).method(), super(cls, self).attr).",
                "url": "https://docs.python.org/3/library/dis.html#opcode-LOAD_SUPER_ATTR"
            };

        case "MAKE_CELL":
            return {
                "html": "<p>Creates a new cell in slot <code>i</code>.  If that slot is nonempty then\nthat value is stored into the new cell.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Creates a new cell in slot i.  If that slot is nonempty then\nthat value is stored into the new cell.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAKE_CELL"
            };

        case "MAKE_FUNCTION":
            return {
                "html": "<p>Pushes a new function object on the stack.  From bottom to top, the consumed\nstack must consist of values if the argument carries a specified flag value</p>\n<ul>\n<li><p><code>0x01</code> a tuple of default values for positional-only and\npositional-or-keyword parameters in positional order</p></li>\n<li><p><code>0x02</code> a dictionary of keyword-only parameters\u2019 default values</p></li>\n<li><p><code>0x04</code> a tuple of strings containing parameters\u2019 annotations</p></li>\n<li><p><code>0x08</code> a tuple containing cells for free variables, making a closure</p></li>\n<li><p>the code associated with the function (at <code>STACK[-1]</code>)</p></li>\n</ul>\n\n<p>Changed in version 3.10: Flag value <code>0x04</code> is a tuple of strings instead of dictionary</p>\n\n\n<p>Changed in version 3.11: Qualified name at <code>STACK[-1]</code> was removed.</p>\n\n",
                "tooltip": "Pushes a new function object on the stack.  From bottom to top, the consumed\nstack must consist of values if the argument carries a specified flag value",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAKE_FUNCTION"
            };

        case "MAP_ADD":
            return {
                "html": "<p>Implements:</p>\n<pre>value = STACK.pop()\nkey = STACK.pop()\ndict.__setitem__(STACK[-i], key, value)\n</pre>\n\n<p>Used to implement dict comprehensions.</p>\n\n<p>Added in version 3.1.</p>\n\n\n<p>Changed in version 3.8: Map value is <code>STACK[-1]</code> and map key is <code>STACK[-2]</code>. Before, those\nwere reversed.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MAP_ADD"
            };

        case "MATCH_CLASS":
            return {
                "html": "<p><code>STACK[-1]</code> is a tuple of keyword attribute names, <code>STACK[-2]</code> is the class\nbeing matched against, and <code>STACK[-3]</code> is the match subject.  count is the\nnumber of positional sub-patterns.</p>\n<p>Pop <code>STACK[-1]</code>, <code>STACK[-2]</code>, and <code>STACK[-3]</code>. If <code>STACK[-3]</code> is an\ninstance of <code>STACK[-2]</code> and has the positional and keyword attributes\nrequired by count and <code>STACK[-1]</code>, push a tuple of extracted attributes.\nOtherwise, push <code>None</code>.</p>\n\n<p>Added in version 3.10.</p>\n\n\n<p>Changed in version 3.11: Previously, this instruction also pushed a boolean value indicating\nsuccess (<code>True</code>) or failure (<code>False</code>).</p>\n\n",
                "tooltip": "STACK[-1] is a tuple of keyword attribute names, STACK[-2] is the class\nbeing matched against, and STACK[-3] is the match subject.  count is the\nnumber of positional sub-patterns.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_CLASS"
            };

        case "MATCH_KEYS":
            return {
                "html": "<p><code>STACK[-1]</code> is a tuple of mapping keys, and <code>STACK[-2]</code> is the match subject.\nIf <code>STACK[-2]</code> contains all of the keys in <code>STACK[-1]</code>, push a <code>tuple</code>\ncontaining the corresponding values. Otherwise, push <code>None</code>.</p>\n\n<p>Added in version 3.10.</p>\n\n\n<p>Changed in version 3.11: Previously, this instruction also pushed a boolean value indicating\nsuccess (<code>True</code>) or failure (<code>False</code>).</p>\n\n",
                "tooltip": "STACK[-1] is a tuple of mapping keys, and STACK[-2] is the match subject.\nIf STACK[-2] contains all of the keys in STACK[-1], push a tuple\ncontaining the corresponding values. Otherwise, push None.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_KEYS"
            };

        case "MATCH_MAPPING":
            return {
                "html": "<p>If <code>STACK[-1]</code> is an instance of <code>collections.abc.Mapping</code> (or, more\ntechnically: if it has the <code>Py_TPFLAGS_MAPPING</code> flag set in its\n<code>tp_flags</code>), push <code>True</code> onto the stack.  Otherwise,\npush <code>False</code>.</p>\n\n<p>Added in version 3.10.</p>\n\n",
                "tooltip": "If STACK[-1] is an instance of collections.abc.Mapping (or, more\ntechnically: if it has the Py_TPFLAGS_MAPPING flag set in its\ntp_flags), push True onto the stack.  Otherwise,\npush False.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_MAPPING"
            };

        case "MATCH_SEQUENCE":
            return {
                "html": "<p>If <code>STACK[-1]</code> is an instance of <code>collections.abc.Sequence</code> and is not an instance\nof <code>str</code>/<code>bytes</code>/<code>bytearray</code> (or, more technically: if it has\nthe <code>Py_TPFLAGS_SEQUENCE</code> flag set in its <code>tp_flags</code>),\npush <code>True</code> onto the stack.  Otherwise, push <code>False</code>.</p>\n\n<p>Added in version 3.10.</p>\n\n",
                "tooltip": "If STACK[-1] is an instance of collections.abc.Sequence and is not an instance\nof str/bytes/bytearray (or, more technically: if it has\nthe Py_TPFLAGS_SEQUENCE flag set in its tp_flags),\npush True onto the stack.  Otherwise, push False.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-MATCH_SEQUENCE"
            };

        case "NOP":
            return {
                "html": "<p>Do nothing code.  Used as a placeholder by the bytecode optimizer, and to\ngenerate line tracing events.</p>\n",
                "tooltip": "Do nothing code.  Used as a placeholder by the bytecode optimizer, and to\ngenerate line tracing events.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-NOP"
            };

        case "POP_BLOCK":
            return {
                "html": "<p>Marks the end of the code block associated with the last <code>SETUP_FINALLY</code>,\n<code>SETUP_CLEANUP</code> or <code>SETUP_WITH</code>.</p>\n",
                "tooltip": "Marks the end of the code block associated with the last SETUP_FINALLY,\nSETUP_CLEANUP or SETUP_WITH.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_BLOCK"
            };

        case "POP_EXCEPT":
            return {
                "html": "<p>Pops a value from the stack, which is used to restore the exception state.</p>\n\n<p>Changed in version 3.11: Exception representation on the stack now consist of one, not three, items.</p>\n\n",
                "tooltip": "Pops a value from the stack, which is used to restore the exception state.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_EXCEPT"
            };

        case "POP_JUMP_IF_FALSE":
            return {
                "html": "<p>If <code>STACK[-1]</code> is false, increments the bytecode counter by delta.\n<code>STACK[-1]</code> is popped.</p>\n\n<p>Changed in version 3.11: The oparg is now a relative delta rather than an absolute target.\nThis opcode is a pseudo-instruction, replaced in final bytecode by\nthe directed versions (forward/backward).</p>\n\n\n<p>Changed in version 3.12: This is no longer a pseudo-instruction.</p>\n\n",
                "tooltip": "If STACK[-1] is false, increments the bytecode counter by delta.\nSTACK[-1] is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_IF_FALSE"
            };

        case "POP_JUMP_IF_NONE":
            return {
                "html": "<p>If <code>STACK[-1]</code> is <code>None</code>, increments the bytecode counter by delta.\n<code>STACK[-1]</code> is popped.</p>\n<p>This opcode is a pseudo-instruction, replaced in final bytecode by\nthe directed versions (forward/backward).</p>\n\n<p>Added in version 3.11.</p>\n\n\n<p>Changed in version 3.12: This is no longer a pseudo-instruction.</p>\n\n",
                "tooltip": "If STACK[-1] is None, increments the bytecode counter by delta.\nSTACK[-1] is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_IF_NONE"
            };

        case "POP_JUMP_IF_NOT_NONE":
            return {
                "html": "<p>If <code>STACK[-1]</code> is not <code>None</code>, increments the bytecode counter by delta.\n<code>STACK[-1]</code> is popped.</p>\n<p>This opcode is a pseudo-instruction, replaced in final bytecode by\nthe directed versions (forward/backward).</p>\n\n<p>Added in version 3.11.</p>\n\n\n<p>Changed in version 3.12: This is no longer a pseudo-instruction.</p>\n\n",
                "tooltip": "If STACK[-1] is not None, increments the bytecode counter by delta.\nSTACK[-1] is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_IF_NOT_NONE"
            };

        case "POP_JUMP_IF_TRUE":
            return {
                "html": "<p>If <code>STACK[-1]</code> is true, increments the bytecode counter by delta.\n<code>STACK[-1]</code> is popped.</p>\n\n<p>Changed in version 3.11: The oparg is now a relative delta rather than an absolute target.\nThis opcode is a pseudo-instruction, replaced in final bytecode by\nthe directed versions (forward/backward).</p>\n\n\n<p>Changed in version 3.12: This is no longer a pseudo-instruction.</p>\n\n",
                "tooltip": "If STACK[-1] is true, increments the bytecode counter by delta.\nSTACK[-1] is popped.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_JUMP_IF_TRUE"
            };

        case "POP_TOP":
            return {
                "html": "<p>Removes the top-of-stack item:</p>\n<pre>STACK.pop()\n</pre>\n\n",
                "tooltip": "Removes the top-of-stack item",
                "url": "https://docs.python.org/3/library/dis.html#opcode-POP_TOP"
            };

        case "PUSH_EXC_INFO":
            return {
                "html": "<p>Pops a value from the stack. Pushes the current exception to the top of the stack.\nPushes the value originally popped back to the stack.\nUsed in exception handlers.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Pops a value from the stack. Pushes the current exception to the top of the stack.\nPushes the value originally popped back to the stack.\nUsed in exception handlers.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PUSH_EXC_INFO"
            };

        case "PUSH_NULL":
            return {
                "html": "<p>Pushes a <code>NULL</code> to the stack.\nUsed in the call sequence to match the <code>NULL</code> pushed by\n<code>LOAD_METHOD</code> for non-method calls.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Pushes a NULL to the stack.\nUsed in the call sequence to match the NULL pushed by\nLOAD_METHOD for non-method calls.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-PUSH_NULL"
            };

        case "RAISE_VARARGS":
            return {
                "html": "<p>Raises an exception using one of the 3 forms of the <code>raise</code> statement,\ndepending on the value of argc:</p>\n<ul>\n<li><p>0: <code>raise</code> (re-raise previous exception)</p></li>\n<li><p>1: <code>raise STACK[-1]</code> (raise exception instance or type at <code>STACK[-1]</code>)</p></li>\n<li><p>2: <code>raise STACK[-2] from STACK[-1]</code> (raise exception instance or type at\n<code>STACK[-2]</code> with <code>__cause__</code> set to <code>STACK[-1]</code>)</p></li>\n</ul>\n",
                "tooltip": "Raises an exception using one of the 3 forms of the raise statement,\ndepending on the value of argc",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RAISE_VARARGS"
            };

        case "RERAISE":
            return {
                "html": "<p>Re-raises the exception currently on top of the stack. If oparg is non-zero,\npops an additional value from the stack which is used to set\n<code>f_lasti</code> of the current frame.</p>\n\n<p>Added in version 3.9.</p>\n\n\n<p>Changed in version 3.11: Exception representation on the stack now consist of one, not three, items.</p>\n\n",
                "tooltip": "Re-raises the exception currently on top of the stack. If oparg is non-zero,\npops an additional value from the stack which is used to set\nf_lasti of the current frame.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RERAISE"
            };

        case "RESUME":
            return {
                "html": "<p>A no-op. Performs internal tracing, debugging and optimization checks.</p>\n<p>The <code>where</code> operand marks where the <code>RESUME</code> occurs:</p>\n<ul>\n<li><p><code>0</code> The start of a function, which is neither a generator, coroutine\nnor an async generator</p></li>\n<li><p><code>1</code> After a <code>yield</code> expression</p></li>\n<li><p><code>2</code> After a <code>yield from</code> expression</p></li>\n<li><p><code>3</code> After an <code>await</code> expression</p></li>\n</ul>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "A no-op. Performs internal tracing, debugging and optimization checks.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RESUME"
            };

        case "RETURN_CONST":
            return {
                "html": "<p>Returns with <code>co_consts[consti]</code> to the caller of the function.</p>\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Returns with co_consts[consti] to the caller of the function.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RETURN_CONST"
            };

        case "RETURN_GENERATOR":
            return {
                "html": "<p>Create a generator, coroutine, or async generator from the current frame.\nUsed as first opcode of in code object for the above mentioned callables.\nClear the current frame and return the newly created generator.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Create a generator, coroutine, or async generator from the current frame.\nUsed as first opcode of in code object for the above mentioned callables.\nClear the current frame and return the newly created generator.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RETURN_GENERATOR"
            };

        case "RETURN_VALUE":
            return {
                "html": "<p>Returns with <code>STACK[-1]</code> to the caller of the function.</p>\n",
                "tooltip": "Returns with STACK[-1] to the caller of the function.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-RETURN_VALUE"
            };

        case "SEND":
            return {
                "html": "<p>Equivalent to <code>STACK[-1] = STACK[-2].send(STACK[-1])</code>. Used in <code>yield from</code>\nand <code>await</code> statements.</p>\n<p>If the call raises <code>StopIteration</code>, pop the top value from the stack,\npush the exception\u2019s <code>value</code> attribute, and increment the bytecode counter\nby delta.</p>\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Equivalent to STACK[-1] = STACK[-2].send(STACK[-1]). Used in yield from\nand await statements.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SEND"
            };

        case "SETUP_ANNOTATIONS":
            return {
                "html": "<p>Checks whether <code>__annotations__</code> is defined in <code>locals()</code>, if not it is\nset up to an empty <code>dict</code>. This opcode is only emitted if a class\nor module body contains variable annotations\nstatically.</p>\n\n<p>Added in version 3.6.</p>\n\n",
                "tooltip": "Checks whether __annotations__ is defined in locals(), if not it is\nset up to an empty dict. This opcode is only emitted if a class\nor module body contains variable annotations\nstatically.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SETUP_ANNOTATIONS"
            };

        case "SETUP_CLEANUP":
            return {
                "html": "<p>Like <code>SETUP_FINALLY</code>, but in case of an exception also pushes the last\ninstruction (<code>lasti</code>) to the stack so that <code>RERAISE</code> can restore it.\nIf an exception occurs, the value stack level and the last instruction on\nthe frame are restored to their current state, and control is transferred\nto the exception handler at <code>target</code>.</p>\n",
                "tooltip": "Like SETUP_FINALLY, but in case of an exception also pushes the last\ninstruction (lasti) to the stack so that RERAISE can restore it.\nIf an exception occurs, the value stack level and the last instruction on\nthe frame are restored to their current state, and control is transferred\nto the exception handler at target.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SETUP_CLEANUP"
            };

        case "SETUP_FINALLY":
            return {
                "html": "<p>Set up an exception handler for the following code block. If an exception\noccurs, the value stack level is restored to its current state and control\nis transferred to the exception handler at <code>target</code>.</p>\n",
                "tooltip": "Set up an exception handler for the following code block. If an exception\noccurs, the value stack level is restored to its current state and control\nis transferred to the exception handler at target.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SETUP_FINALLY"
            };

        case "SETUP_WITH":
            return {
                "html": "<p>Like <code>SETUP_CLEANUP</code>, but in case of an exception one more item is popped\nfrom the stack before control is transferred to the exception handler at\n<code>target</code>.</p>\n<p>This variant is used in <code>with</code> and <code>async with</code>\nconstructs, which push the return value of the context manager\u2019s\n<code>__enter__()</code> or <code>__aenter__()</code> to the stack.</p>\n",
                "tooltip": "Like SETUP_CLEANUP, but in case of an exception one more item is popped\nfrom the stack before control is transferred to the exception handler at\ntarget.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SETUP_WITH"
            };

        case "SET_ADD":
            return {
                "html": "<p>Implements:</p>\n<pre>item = STACK.pop()\nset.add(STACK[-i], item)\n</pre>\n\n<p>Used to implement set comprehensions.</p>\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SET_ADD"
            };

        case "SET_UPDATE":
            return {
                "html": "<p>Implements:</p>\n<pre>seq = STACK.pop()\nset.update(STACK[-i], seq)\n</pre>\n\n<p>Used to build sets.</p>\n\n<p>Added in version 3.9.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SET_UPDATE"
            };

        case "STORE_ATTR":
            return {
                "html": "<p>Implements:</p>\n<pre>obj = STACK.pop()\nvalue = STACK.pop()\nobj.name = value\n</pre>\n\n<p>where namei is the index of name in <code>co_names</code> of the\ncode object.</p>\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_ATTR"
            };

        case "STORE_DEREF":
            return {
                "html": "<p>Stores <code>STACK.pop()</code> into the cell contained in slot <code>i</code> of the \u201cfast locals\u201d\nstorage.</p>\n\n<p>Changed in version 3.11: <code>i</code> is no longer offset by the length of <code>co_varnames</code>.</p>\n\n",
                "tooltip": "Stores STACK.pop() into the cell contained in slot i of the \u201cfast locals\u201d\nstorage.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_DEREF"
            };

        case "STORE_FAST":
            return {
                "html": "<p>Stores <code>STACK.pop()</code> into the local <code>co_varnames[var_num]</code>.</p>\n",
                "tooltip": "Stores STACK.pop() into the local co_varnames[var_num].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_FAST"
            };

        case "STORE_GLOBAL":
            return {
                "html": "<p>Works as <code>STORE_NAME</code>, but stores the name as a global.</p>\n",
                "tooltip": "Works as STORE_NAME, but stores the name as a global.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_GLOBAL"
            };

        case "STORE_NAME":
            return {
                "html": "<p>Implements <code>name = STACK.pop()</code>. namei is the index of name in the attribute\n<code>co_names</code> of the code object.\nThe compiler tries to use <code>STORE_FAST</code> or <code>STORE_GLOBAL</code> if possible.</p>\n",
                "tooltip": "Implements name = STACK.pop(). namei is the index of name in the attribute\nco_names of the code object.\nThe compiler tries to use STORE_FAST or STORE_GLOBAL if possible.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_NAME"
            };

        case "STORE_SLICE":
            return {
                "html": "<p>Implements:</p>\n<pre>end = STACK.pop()\nstart = STACK.pop()\ncontainer = STACK.pop()\nvalues = STACK.pop()\ncontainer[start:end] = value\n</pre>\n\n\n<p>Added in version 3.12.</p>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_SLICE"
            };

        case "STORE_SUBSCR":
            return {
                "html": "<p>Implements:</p>\n<pre>key = STACK.pop()\ncontainer = STACK.pop()\nvalue = STACK.pop()\ncontainer[key] = value\n</pre>\n\n",
                "tooltip": "Implements",
                "url": "https://docs.python.org/3/library/dis.html#opcode-STORE_SUBSCR"
            };

        case "SWAP":
            return {
                "html": "<p>Swap the top of the stack with the i-th element:</p>\n<pre>STACK[-i], STACK[-1] = STACK[-1], STACK[-i]\n</pre>\n\n\n<p>Added in version 3.11.</p>\n\n",
                "tooltip": "Swap the top of the stack with the i-th element",
                "url": "https://docs.python.org/3/library/dis.html#opcode-SWAP"
            };

        case "UNARY_INVERT":
            return {
                "html": "<p>Implements <code>STACK[-1] = ~STACK[-1]</code>.</p>\n",
                "tooltip": "Implements STACK[-1] = ~STACK[-1].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_INVERT"
            };

        case "UNARY_NEGATIVE":
            return {
                "html": "<p>Implements <code>STACK[-1] = -STACK[-1]</code>.</p>\n",
                "tooltip": "Implements STACK[-1] = -STACK[-1].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_NEGATIVE"
            };

        case "UNARY_NOT":
            return {
                "html": "<p>Implements <code>STACK[-1] = not STACK[-1]</code>.</p>\n",
                "tooltip": "Implements STACK[-1] = not STACK[-1].",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNARY_NOT"
            };

        case "UNPACK_EX":
            return {
                "html": "<p>Implements assignment with a starred target: Unpacks an iterable in <code>STACK[-1]</code>\ninto individual values, where the total number of values can be smaller than the\nnumber of items in the iterable: one of the new values will be a list of all\nleftover items.</p>\n<p>The number of values before and after the list value is limited to 255.</p>\n<p>The number of values before the list value is encoded in the argument of the\nopcode. The number of values after the list if any is encoded using an\n<code>EXTENDED_ARG</code>. As a consequence, the argument can be seen as a two bytes values\nwhere the low byte of counts is the number of values before the list value, the\nhigh byte of counts the number of values after it.</p>\n<p>The extracted values are put onto the stack right-to-left, i.e. <code>a, *b, c = d</code>\nwill be stored after execution as <code>STACK.extend((a, b, c))</code>.</p>\n",
                "tooltip": "Implements assignment with a starred target: Unpacks an iterable in STACK[-1]\ninto individual values, where the total number of values can be smaller than the\nnumber of items in the iterable: one of the new values will be a list of all\nleftover items.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNPACK_EX"
            };

        case "UNPACK_SEQUENCE":
            return {
                "html": "<p>Unpacks <code>STACK[-1]</code> into count individual values, which are put onto the stack\nright-to-left. Require there to be exactly count values.:</p>\n<pre>assert(len(STACK[-1]) == count)\nSTACK.extend(STACK.pop()[:-count-1:-1])\n</pre>\n\n",
                "tooltip": "Unpacks STACK[-1] into count individual values, which are put onto the stack\nright-to-left. Require there to be exactly count values.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-UNPACK_SEQUENCE"
            };

        case "WITH_EXCEPT_START":
            return {
                "html": "<p>Calls the function in position 4 on the stack with arguments (type, val, tb)\nrepresenting the exception at the top of the stack.\nUsed to implement the call <code>context_manager.__exit__(*exc_info())</code> when an exception\nhas occurred in a <code>with</code> statement.</p>\n\n<p>Added in version 3.9.</p>\n\n\n<p>Changed in version 3.11: The <code>__exit__</code> function is in position 4 of the stack rather than 7.\nException representation on the stack now consist of one, not three, items.</p>\n\n",
                "tooltip": "Calls the function in position 4 on the stack with arguments (type, val, tb)\nrepresenting the exception at the top of the stack.\nUsed to implement the call context_manager.__exit__(*exc_info()) when an exception\nhas occurred in a with statement.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-WITH_EXCEPT_START"
            };

        case "YIELD_VALUE":
            return {
                "html": "<p>Yields <code>STACK.pop()</code> from a generator.</p>\n\n<p>Changed in version 3.11: oparg set to be the stack depth.</p>\n\n\n<p>Changed in version 3.12: oparg set to be the exception block depth, for efficient closing of generators.</p>\n\n",
                "tooltip": "Yields STACK.pop() from a generator.",
                "url": "https://docs.python.org/3/library/dis.html#opcode-YIELD_VALUE"
            };


    }
}
