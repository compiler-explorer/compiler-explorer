export function getAsmOpcode(opcode) {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case 'AALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aaload`,
                html: `<p>Instruction aaload: Load reference from array </p><p>Format: aaload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.aaload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">reference</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">reference</code> <span class="emphasis"><em>value</em></span> in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved and pushed onto the operand stack. </p>`,
                tooltip: `Load reference from array `,
            };
        case 'AASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aastore`,
                html: `<p>Instruction aastore: Store into reference array </p><p>Format: aastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.aastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">reference</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>, and <span class="emphasis"><em>value</em></span> must be of type <code class="literal">reference</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. </p>`,
                tooltip: `Store into reference array `,
            };
        case 'ACONST_NULL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aconst_null`,
                html: `<p>Instruction aconst_null: Push null</p><p>Format: aconst_null</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <code class="literal">null</code></p><p><a name="jvms-6.5.aconst_null.desc-100"></a> Push the <code class="literal">null</code> object <code class="literal">reference</code> onto the operand stack. </p>`,
                tooltip: `Push null`,
            };
        case 'ALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aload`,
                html: `<p>Instruction aload: Load reference from local variable </p><p>Format: aload index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.aload.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at <span class="emphasis"><em>index</em></span> must contain a <code class="literal">reference</code>. The <span class="emphasis"><em>objectref</em></span> in the local variable at <span class="emphasis"><em>index</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load reference from local variable `,
            };
        case 'ALOAD_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aload_n`,
                html: `<p>Instruction aload_0: Load reference from local variable </p><p>Format: aload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.aload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">reference</code>. The <span class="emphasis"><em>objectref</em></span> in the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load reference from local variable `,
            };
        case 'ALOAD_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aload_n`,
                html: `<p>Instruction aload_1: Load reference from local variable </p><p>Format: aload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.aload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">reference</code>. The <span class="emphasis"><em>objectref</em></span> in the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load reference from local variable `,
            };
        case 'ALOAD_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aload_n`,
                html: `<p>Instruction aload_2: Load reference from local variable </p><p>Format: aload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.aload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">reference</code>. The <span class="emphasis"><em>objectref</em></span> in the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load reference from local variable `,
            };
        case 'ALOAD_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.aload_n`,
                html: `<p>Instruction aload_3: Load reference from local variable </p><p>Format: aload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.aload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">reference</code>. The <span class="emphasis"><em>objectref</em></span> in the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load reference from local variable `,
            };
        case 'ANEWARRAY':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.anewarray`,
                html: `<p>Instruction anewarray: Create new array of reference</p><p>Format: anewarray indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>count</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>arrayref</em></span></p><p><a name="jvms-6.5.anewarray.desc-100"></a> The <span class="emphasis"><em>count</em></span> must be of type <code class="literal">int</code>. It is popped off the operand stack. The <span class="emphasis"><em>count</em></span> represents the number of components of the array to be created. The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a class, array, or interface type. The named class, array, or interface type is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.1" title="5.4.3.1.&nbsp;Class and Interface Resolution">§5.4.3.1</a>). A new array with components of that type, of length <span class="emphasis"><em>count</em></span>, is allocated from the garbage-collected heap, and a <code class="literal">reference</code> <span class="emphasis"><em>arrayref</em></span> to this new array object is pushed onto the operand stack. All components of the new array are initialized to <code class="literal">null</code>, the default value for <code class="literal">reference</code> types (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.4" title="2.4.&nbsp;Reference Types and Values">§2.4</a>). </p>`,
                tooltip: `Create new array of reference`,
            };
        case 'ARETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.areturn`,
                html: `<p>Instruction areturn: Return reference from method </p><p>Format: areturn</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.areturn.desc-100"></a> The <span class="emphasis"><em>objectref</em></span> must be of type <code class="literal">reference</code> and must refer to an object of a type that is assignment compatible (JLS §5.2) with the type represented by the return descriptor (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-4.html#jvms-4.3.3" title="4.3.3.&nbsp;Method Descriptors">§4.3.3</a>) of the current method. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, <span class="emphasis"><em>objectref</em></span> is popped from the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) and pushed onto the operand stack of the frame of the invoker. Any other values on the operand stack of the current method are discarded. </p>`,
                tooltip: `Return reference from method `,
            };
        case 'ARRAYLENGTH':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.arraylength`,
                html: `<p>Instruction arraylength: Get length of array</p><p>Format: arraylength</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>length</em></span></p><p><a name="jvms-6.5.arraylength.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array. It is popped from the operand stack. The <span class="emphasis"><em>length</em></span> of the array it references is determined. That <span class="emphasis"><em>length</em></span> is pushed onto the operand stack as an <code class="literal">int</code>. </p>`,
                tooltip: `Get length of array`,
            };
        case 'ASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.astore`,
                html: `<p>Instruction astore: Store reference into local variable </p><p>Format: astore index</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.astore.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>objectref</em></span> on the top of the operand stack must be of type <code class="literal">returnAddress</code> or of type <code class="literal">reference</code>. It is popped from the operand stack, and the value of the local variable at <span class="emphasis"><em>index</em></span> is set to <span class="emphasis"><em>objectref</em></span>. </p>`,
                tooltip: `Store reference into local variable `,
            };
        case 'ASTORE_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.astore_n`,
                html: `<p>Instruction astore_0: Store reference into local variable </p><p>Format: astore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.astore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>objectref</em></span> on the top of the operand stack must be of type <code class="literal">returnAddress</code> or of type <code class="literal">reference</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>objectref</em></span>. </p>`,
                tooltip: `Store reference into local variable `,
            };
        case 'ASTORE_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.astore_n`,
                html: `<p>Instruction astore_1: Store reference into local variable </p><p>Format: astore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.astore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>objectref</em></span> on the top of the operand stack must be of type <code class="literal">returnAddress</code> or of type <code class="literal">reference</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>objectref</em></span>. </p>`,
                tooltip: `Store reference into local variable `,
            };
        case 'ASTORE_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.astore_n`,
                html: `<p>Instruction astore_2: Store reference into local variable </p><p>Format: astore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.astore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>objectref</em></span> on the top of the operand stack must be of type <code class="literal">returnAddress</code> or of type <code class="literal">reference</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>objectref</em></span>. </p>`,
                tooltip: `Store reference into local variable `,
            };
        case 'ASTORE_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.astore_n`,
                html: `<p>Instruction astore_3: Store reference into local variable </p><p>Format: astore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.astore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>objectref</em></span> on the top of the operand stack must be of type <code class="literal">returnAddress</code> or of type <code class="literal">reference</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>objectref</em></span>. </p>`,
                tooltip: `Store reference into local variable `,
            };
        case 'ATHROW':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.athrow`,
                html: `<p>Instruction athrow: Throw exception or error</p><p>Format: athrow</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.athrow.desc-100"></a> The <span class="emphasis"><em>objectref</em></span> must be of type <code class="literal">reference</code> and must refer to an object that is an instance of class <code class="literal">Throwable</code> or of a subclass of <code class="literal">Throwable</code>. It is popped from the operand stack. The <span class="emphasis"><em>objectref</em></span> is then thrown by searching the current method (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) for the first exception handler that matches the class of <span class="emphasis"><em>objectref</em></span>, as given by the algorithm in <a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.10" title="2.10.&nbsp;Exceptions">§2.10</a>. </p>`,
                tooltip: `Throw exception or error`,
            };
        case 'BALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.baload`,
                html: `<p>Instruction baload: Load byte or boolean from array </p><p>Format: baload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.baload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">byte</code> or of type <code class="literal">boolean</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">byte</code> <span class="emphasis"><em>value</em></span> in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved, sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>value</em></span>, and pushed onto the top of the operand stack. </p>`,
                tooltip: `Load byte or boolean from array `,
            };
        case 'BASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.bastore`,
                html: `<p>Instruction bastore: Store into byte or boolean array </p><p>Format: bastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.bastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">byte</code> or of type <code class="literal">boolean</code>. The <span class="emphasis"><em>index</em></span> and the <span class="emphasis"><em>value</em></span> must both be of type <code class="literal">int</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. </p>`,
                tooltip: `Store into byte or boolean array `,
            };
        case 'BIPUSH':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.bipush`,
                html: `<p>Instruction bipush: Push byte</p><p>Format: bipush byte</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.bipush.desc-100"></a> The immediate <span class="emphasis"><em>byte</em></span> is sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>value</em></span>. That <span class="emphasis"><em>value</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Push byte`,
            };
        case 'CALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.caload`,
                html: `<p>Instruction caload: Load char from array </p><p>Format: caload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.caload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">char</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The component of the array at <span class="emphasis"><em>index</em></span> is retrieved and zero-extended to an <code class="literal">int</code> <span class="emphasis"><em>value</em></span>. That <span class="emphasis"><em>value</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load char from array `,
            };
        case 'CASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.castore`,
                html: `<p>Instruction castore: Store into char array </p><p>Format: castore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.castore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">char</code>. The <span class="emphasis"><em>index</em></span> and the <span class="emphasis"><em>value</em></span> must both be of type <code class="literal">int</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>value</em></span> is truncated to a <code class="literal">char</code> and stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into char array `,
            };
        case 'CHECKCAST':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.checkcast`,
                html: `<p>Instruction checkcast: Check whether object is of given type</p><p>Format: checkcast indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.checkcast.desc-100"></a> The <span class="emphasis"><em>objectref</em></span> must be of type <code class="literal">reference</code>. The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a class, array, or interface type. </p>`,
                tooltip: `Check whether object is of given type`,
            };
        case 'D2F':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.d2f`,
                html: `<p>Instruction d2f: Convert double to float</p><p>Format: d2f</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.d2f.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>) resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to a <code class="literal">float</code> <span class="emphasis"><em>result</em></span> using the round to nearest rounding policy (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8" title="2.8.&nbsp;Floating-Point Arithmetic">§2.8</a>). The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert double to float`,
            };
        case 'D2I':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.d2i`,
                html: `<p>Instruction d2i: Convert double to int</p><p>Format: d2i</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.d2i.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>) resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack: </p>`,
                tooltip: `Convert double to int`,
            };
        case 'D2L':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.d2l`,
                html: `<p>Instruction d2l: Convert double to long</p><p>Format: d2l</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.d2l.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>) resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to a <code class="literal">long</code>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack: </p>`,
                tooltip: `Convert double to long`,
            };
        case 'DADD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dadd`,
                html: `<p>Instruction dadd: Add double</p><p>Format: dadd</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dadd.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' + <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Add double`,
            };
        case 'DALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.daload`,
                html: `<p>Instruction daload: Load double from array </p><p>Format: daload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.daload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">double</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">double</code> <span class="emphasis"><em>value</em></span> in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved and pushed onto the operand stack. </p>`,
                tooltip: `Load double from array `,
            };
        case 'DASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dastore`,
                html: `<p>Instruction dastore: Store into double array </p><p>Format: dastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">double</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>, and value must be of type <code class="literal">double</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">double</code> <span class="emphasis"><em>value</em></span> undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>', which is stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into double array `,
            };
        case 'DCMPG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dcmp_op`,
                html: `<p>Instruction dcmpg: Compare double</p><p>Format: dcmp[op]</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dcmp_op.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. A floating-point comparison is performed: </p>`,
                tooltip: `Compare double`,
            };
        case 'DCMPL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dcmp_op`,
                html: `<p>Instruction dcmpl: Compare double</p><p>Format: dcmp[op]</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dcmp_op.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. A floating-point comparison is performed: </p>`,
                tooltip: `Compare double`,
            };
        case 'DCONST_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dconst_d`,
                html: `<p>Instruction dconst_0: Push double</p><p>Format: dconst_[d]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>d</em></span>&gt; </p><p><a name="jvms-6.5.dconst_d.desc-100"></a> Push the <code class="literal">double</code> constant &lt;<span class="emphasis"><em>d</em></span>&gt; (0.0 or 1.0) onto the operand stack. </p>`,
                tooltip: `Push double`,
            };
        case 'DCONST_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dconst_d`,
                html: `<p>Instruction dconst_1: Push double</p><p>Format: dconst_[d]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>d</em></span>&gt; </p><p><a name="jvms-6.5.dconst_d.desc-100"></a> Push the <code class="literal">double</code> constant &lt;<span class="emphasis"><em>d</em></span>&gt; (0.0 or 1.0) onto the operand stack. </p>`,
                tooltip: `Push double`,
            };
        case 'DDIV':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ddiv`,
                html: `<p>Instruction ddiv: Divide double</p><p>Format: ddiv</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ddiv.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' / <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Divide double`,
            };
        case 'DLOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dload`,
                html: `<p>Instruction dload: Load double from local variable </p><p>Format: dload index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dload.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at <span class="emphasis"><em>index</em></span> must contain a <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> of the local variable at <span class="emphasis"><em>index</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load double from local variable `,
            };
        case 'DLOAD_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dload_n`,
                html: `<p>Instruction dload_0: Load double from local variable </p><p>Format: dload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load double from local variable `,
            };
        case 'DLOAD_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dload_n`,
                html: `<p>Instruction dload_1: Load double from local variable </p><p>Format: dload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load double from local variable `,
            };
        case 'DLOAD_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dload_n`,
                html: `<p>Instruction dload_2: Load double from local variable </p><p>Format: dload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load double from local variable `,
            };
        case 'DLOAD_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dload_n`,
                html: `<p>Instruction dload_3: Load double from local variable </p><p>Format: dload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load double from local variable `,
            };
        case 'DMUL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dmul`,
                html: `<p>Instruction dmul: Multiply double</p><p>Format: dmul</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dmul.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' * <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Multiply double`,
            };
        case 'DNEG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dneg`,
                html: `<p>Instruction dneg: Negate double</p><p>Format: dneg</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dneg.desc-100"></a> The value must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is the arithmetic negation of <span class="emphasis"><em>value</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Negate double`,
            };
        case 'DREM':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.drem`,
                html: `<p>Instruction drem: Remainder double</p><p>Format: drem</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.drem.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is calculated and pushed onto the operand stack. </p>`,
                tooltip: `Remainder double`,
            };
        case 'DRETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dreturn`,
                html: `<p>Instruction dreturn: Return double from method </p><p>Format: dreturn</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.dreturn.desc-100"></a> The current method must have return type <code class="literal">double</code>. The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">double</code>. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, <span class="emphasis"><em>value</em></span> is popped from the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The <span class="emphasis"><em>value</em></span>' is pushed onto the operand stack of the frame of the invoker. Any other values on the operand stack of the current method are discarded. </p>`,
                tooltip: `Return double from method `,
            };
        case 'DSTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dstore`,
                html: `<p>Instruction dstore: Store double into local variable </p><p>Format: dstore index</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dstore.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The local variables at <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 are set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store double into local variable `,
            };
        case 'DSTORE_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dstore_n`,
                html: `<p>Instruction dstore_0: Store double into local variable </p><p>Format: dstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store double into local variable `,
            };
        case 'DSTORE_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dstore_n`,
                html: `<p>Instruction dstore_1: Store double into local variable </p><p>Format: dstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store double into local variable `,
            };
        case 'DSTORE_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dstore_n`,
                html: `<p>Instruction dstore_2: Store double into local variable </p><p>Format: dstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store double into local variable `,
            };
        case 'DSTORE_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dstore_n`,
                html: `<p>Instruction dstore_3: Store double into local variable </p><p>Format: dstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.dstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">double</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store double into local variable `,
            };
        case 'DSUB':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dsub`,
                html: `<p>Instruction dsub: Subtract double</p><p>Format: dsub</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.dsub.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">double</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">double</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' - <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Subtract double`,
            };
        case 'DUP':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup`,
                html: `<p>Instruction dup: Duplicate the top operand stack value</p><p>Format: dup</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span>, <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.dup.desc-100"></a> Duplicate the top value on the operand stack and push the duplicated value onto the operand stack. </p>`,
                tooltip: `Duplicate the top operand stack value`,
            };
        case 'DUP_X1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup_x1`,
                html: `<p>Instruction dup_x1: Duplicate the top operand stack value and insert two values down</p><p>Format: dup_x1</p><p>Operand Stack: ..., <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span></p><p><a name="jvms-6.5.dup_x1.desc-100"></a> Duplicate the top value on the operand stack and insert the duplicated value two values down in the operand stack. </p>`,
                tooltip: `Duplicate the top operand stack value and insert two values down`,
            };
        case 'DUP_X2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup_x2`,
                html: `<p>Instruction dup_x2: Duplicate the top operand stack value and insert two or three values down</p><p>Format: dup_x2</p><p>Operand Stack: <a name="jvms-6.5.dup_x2.stack-100"></a>Form 1: <a name="jvms-6.5.dup_x2.stack-100-A"></a>..., <span class="emphasis"><em>value3</em></span>, <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span></p><p><a name="jvms-6.5.dup_x2.desc-100"></a> Duplicate the top value on the operand stack and insert the duplicated value two or three values down in the operand stack. </p>`,
                tooltip: `Duplicate the top operand stack value and insert two or three values down`,
            };
        case 'DUP2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup2`,
                html: `<p>Instruction dup2: Duplicate the top one or two operand stack values</p><p>Format: dup2</p><p>Operand Stack: <a name="jvms-6.5.dup2.stack-100"></a>Form 1: <a name="jvms-6.5.dup2.stack-100-A"></a>..., <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span></p><p><a name="jvms-6.5.dup2.desc-100"></a> Duplicate the top one or two values on the operand stack and push the duplicated value or values back onto the operand stack in the original order. </p>`,
                tooltip: `Duplicate the top one or two operand stack values`,
            };
        case 'DUP2_X1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup2_x1`,
                html: `<p>Instruction dup2_x1: Duplicate the top one or two operand stack values and insert two or three values down</p><p>Format: dup2_x1</p><p>Operand Stack: <a name="jvms-6.5.dup2_x1.stack-100"></a>Form 1: <a name="jvms-6.5.dup2_x1.stack-100-A"></a>..., <span class="emphasis"><em>value3</em></span>, <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span></p><p><a name="jvms-6.5.dup2_x1.desc-100"></a> Duplicate the top one or two values on the operand stack and insert the duplicated values, in the original order, one value beneath the original value or values in the operand stack. </p>`,
                tooltip: `Duplicate the top one or two operand stack values and insert two or three values down`,
            };
        case 'DUP2_X2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.dup2_x2`,
                html: `<p>Instruction dup2_x2: Duplicate the top one or two operand stack values and insert two, three, or four values down</p><p>Format: dup2_x2</p><p>Operand Stack: <a name="jvms-6.5.dup2_x2.stack-100"></a>Form 1: <a name="jvms-6.5.dup2_x2.stack-100-A"></a>..., <span class="emphasis"><em>value4</em></span>, <span class="emphasis"><em>value3</em></span>, <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span></p><p><a name="jvms-6.5.dup2_x2.desc-100"></a> Duplicate the top one or two values on the operand stack and insert the duplicated values, in the original order, into the operand stack. </p>`,
                tooltip: `Duplicate the top one or two operand stack values and insert two, three, or four values down`,
            };
        case 'F2D':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.f2d`,
                html: `<p>Instruction f2d: Convert float to double</p><p>Format: f2d</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.f2d.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to a <code class="literal">double</code> <span class="emphasis"><em>result</em></span>. This <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert float to double`,
            };
        case 'F2I':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.f2i`,
                html: `<p>Instruction f2i: Convert float to int</p><p>Format: f2i</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.f2i.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span>. This <span class="emphasis"><em>result</em></span> is pushed onto the operand stack: </p>`,
                tooltip: `Convert float to int`,
            };
        case 'F2L':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.f2l`,
                html: `<p>Instruction f2l: Convert float to long</p><p>Format: f2l</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.f2l.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. Then <span class="emphasis"><em>value</em></span>' is converted to a <code class="literal">long</code> <span class="emphasis"><em>result</em></span>. This <span class="emphasis"><em>result</em></span> is pushed onto the operand stack: </p>`,
                tooltip: `Convert float to long`,
            };
        case 'FADD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fadd`,
                html: `<p>Instruction fadd: Add float</p><p>Format: fadd</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fadd.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' + <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Add float`,
            };
        case 'FALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.faload`,
                html: `<p>Instruction faload: Load float from array </p><p>Format: faload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.faload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">float</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">float</code> value in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved and pushed onto the operand stack. </p>`,
                tooltip: `Load float from array `,
            };
        case 'FASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fastore`,
                html: `<p>Instruction fastore: Store into float array </p><p>Format: fastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">float</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>, and the <span class="emphasis"><em>value</em></span> must be of type <code class="literal">float</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">float</code> <span class="emphasis"><em>value</em></span> undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>', and <span class="emphasis"><em>value</em></span>' is stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into float array `,
            };
        case 'FCMPG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fcmp_op`,
                html: `<p>Instruction fcmpg: Compare float</p><p>Format: fcmp[op]</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fcmp_op.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. A floating-point comparison is performed: </p>`,
                tooltip: `Compare float`,
            };
        case 'FCMPL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fcmp_op`,
                html: `<p>Instruction fcmpl: Compare float</p><p>Format: fcmp[op]</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fcmp_op.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. A floating-point comparison is performed: </p>`,
                tooltip: `Compare float`,
            };
        case 'FCONST_0, 1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fconst_f`,
                html: `<p>Instruction fconst_0, 1: Push float</p><p>Format: fconst_[f]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>f</em></span>&gt; </p><p><a name="jvms-6.5.fconst_f.desc-100"></a> Push the <code class="literal">float</code> constant &lt;<span class="emphasis"><em>f</em></span>&gt; (0.0, 1.0, or 2.0) onto the operand stack. </p>`,
                tooltip: `Push float`,
            };
        case 'FCONST_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fconst_f`,
                html: `<p>Instruction fconst_2: Push float</p><p>Format: fconst_[f]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>f</em></span>&gt; </p><p><a name="jvms-6.5.fconst_f.desc-100"></a> Push the <code class="literal">float</code> constant &lt;<span class="emphasis"><em>f</em></span>&gt; (0.0, 1.0, or 2.0) onto the operand stack. </p>`,
                tooltip: `Push float`,
            };
        case 'FDIV':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fdiv`,
                html: `<p>Instruction fdiv: Divide float</p><p>Format: fdiv</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fdiv.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' / <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Divide float`,
            };
        case 'FLOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fload`,
                html: `<p>Instruction fload: Load float from local variable </p><p>Format: fload index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.fload.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at <span class="emphasis"><em>index</em></span> must contain a <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> of the local variable at <span class="emphasis"><em>index</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load float from local variable `,
            };
        case 'FLOAD_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fload_n`,
                html: `<p>Instruction fload_0: Load float from local variable </p><p>Format: fload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.fload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load float from local variable `,
            };
        case 'FLOAD_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fload_n`,
                html: `<p>Instruction fload_1: Load float from local variable </p><p>Format: fload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.fload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load float from local variable `,
            };
        case 'FLOAD_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fload_n`,
                html: `<p>Instruction fload_2: Load float from local variable </p><p>Format: fload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.fload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load float from local variable `,
            };
        case 'FLOAD_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fload_n`,
                html: `<p>Instruction fload_3: Load float from local variable </p><p>Format: fload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.fload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load float from local variable `,
            };
        case 'FMUL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fmul`,
                html: `<p>Instruction fmul: Multiply float</p><p>Format: fmul</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fmul.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' * <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Multiply float`,
            };
        case 'FNEG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fneg`,
                html: `<p>Instruction fneg: Negate float</p><p>Format: fneg</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fneg.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is the arithmetic negation of <span class="emphasis"><em>value</em></span>'. This <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Negate float`,
            };
        case 'FREM':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.frem`,
                html: `<p>Instruction frem: Remainder float</p><p>Format: frem</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.frem.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is calculated and pushed onto the operand stack. </p>`,
                tooltip: `Remainder float`,
            };
        case 'FRETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.freturn`,
                html: `<p>Instruction freturn: Return float from method </p><p>Format: freturn</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.freturn.desc-100"></a> The current method must have return type <code class="literal">float</code>. The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">float</code>. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, <span class="emphasis"><em>value</em></span> is popped from the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The <span class="emphasis"><em>value</em></span>' is pushed onto the operand stack of the frame of the invoker. Any other values on the operand stack of the current method are discarded. </p>`,
                tooltip: `Return float from method `,
            };
        case 'FSTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fstore`,
                html: `<p>Instruction fstore: Store float into local variable </p><p>Format: fstore index</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fstore.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The value of the local variable at <span class="emphasis"><em>index</em></span> is set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store float into local variable `,
            };
        case 'FSTORE_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fstore_n`,
                html: `<p>Instruction fstore_0: Store float into local variable </p><p>Format: fstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fstore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store float into local variable `,
            };
        case 'FSTORE_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fstore_n`,
                html: `<p>Instruction fstore_1: Store float into local variable </p><p>Format: fstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fstore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store float into local variable `,
            };
        case 'FSTORE_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fstore_n`,
                html: `<p>Instruction fstore_2: Store float into local variable </p><p>Format: fstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fstore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store float into local variable `,
            };
        case 'FSTORE_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fstore_n`,
                html: `<p>Instruction fstore_3: Store float into local variable </p><p>Format: fstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.fstore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">float</code>. It is popped from the operand stack and undergoes value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value</em></span>'. The value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>'. </p>`,
                tooltip: `Store float into local variable `,
            };
        case 'FSUB':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.fsub`,
                html: `<p>Instruction fsub: Subtract float</p><p>Format: fsub</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.fsub.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">float</code>. The values are popped from the operand stack and undergo value set conversion (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8.3" title="2.8.3.&nbsp;Value Set Conversion">§2.8.3</a>), resulting in <span class="emphasis"><em>value1</em></span>' and <span class="emphasis"><em>value2</em></span>'. The <code class="literal">float</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span>' - <span class="emphasis"><em>value2</em></span>'. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Subtract float`,
            };
        case 'GETFIELD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.getfield`,
                html: `<p>Instruction getfield: Fetch field from object</p><p>Format: getfield indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.getfield.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a field (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor of the field as well as a symbolic reference to the class in which the field is to be found. The referenced field is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.2" title="5.4.3.2.&nbsp;Field Resolution">§5.4.3.2</a>). </p>`,
                tooltip: `Fetch field from object`,
            };
        case 'GETSTATIC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.getstatic`,
                html: `<p>Instruction getstatic: Get static field from class </p><p>Format: getstatic indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.getstatic.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a field (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor of the field as well as a symbolic reference to the class or interface in which the field is to be found. The referenced field is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.2" title="5.4.3.2.&nbsp;Field Resolution">§5.4.3.2</a>). </p>`,
                tooltip: `Get static field from class `,
            };
        case 'GOTO':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.goto`,
                html: `<p>Instruction goto: Branch always</p><p>Format: goto branchbyte1 branchbyte2</p><p>Operand Stack: No change undefined</p><p><a name="jvms-6.5.goto.desc-100"></a> The unsigned bytes <span class="emphasis"><em>branchbyte1</em></span> and <span class="emphasis"><em>branchbyte2</em></span> are used to construct a signed 16-bit <span class="emphasis"><em>branchoffset</em></span>, where <span class="emphasis"><em>branchoffset</em></span> is (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte2</em></span>. Execution proceeds at that offset from the address of the opcode of this <span class="emphasis"><em>goto</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>goto</em></span> instruction. </p>`,
                tooltip: `Branch always`,
            };
        case 'GOTO_W':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.goto_w`,
                html: `<p>Instruction goto_w: Branch always (wide index)</p><p>Format: goto_w branchbyte1 branchbyte2 branchbyte3 branchbyte4</p><p>Operand Stack: No change undefined</p><p><a name="jvms-6.5.goto_w.desc-100"></a> The unsigned bytes <span class="emphasis"><em>branchbyte1</em></span>, <span class="emphasis"><em>branchbyte2</em></span>, <span class="emphasis"><em>branchbyte3</em></span>, and <span class="emphasis"><em>branchbyte4</em></span> are used to construct a signed 32-bit <span class="emphasis"><em>branchoffset</em></span>, where <span class="emphasis"><em>branchoffset</em></span> is (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 24) | (<span class="emphasis"><em>branchbyte2</em></span> <code class="literal">&lt;&lt;</code> 16) | (<span class="emphasis"><em>branchbyte3</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte4</em></span>. Execution proceeds at that offset from the address of the opcode of this <span class="emphasis"><em>goto_w</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>goto_w</em></span> instruction. </p>`,
                tooltip: `Branch always (wide index)`,
            };
        case 'I2B':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2b`,
                html: `<p>Instruction i2b: Convert int to byte</p><p>Format: i2b</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2b.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, truncated to a <code class="literal">byte</code>, then sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to byte`,
            };
        case 'I2C':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2c`,
                html: `<p>Instruction i2c: Convert int to char</p><p>Format: i2c</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2c.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, truncated to <code class="literal">char</code>, then zero-extended to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to char`,
            };
        case 'I2D':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2d`,
                html: `<p>Instruction i2d: Convert int to double</p><p>Format: i2d</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2d.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack and converted to a <code class="literal">double</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to double`,
            };
        case 'I2F':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2f`,
                html: `<p>Instruction i2f: Convert int to float</p><p>Format: i2f</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2f.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack and converted to the <code class="literal">float</code> <span class="emphasis"><em>result</em></span> using the round to nearest rounding policy (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8" title="2.8.&nbsp;Floating-Point Arithmetic">§2.8</a>). The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to float`,
            };
        case 'I2L':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2l`,
                html: `<p>Instruction i2l: Convert int to long</p><p>Format: i2l</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2l.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack and sign-extended to a <code class="literal">long</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to long`,
            };
        case 'I2S':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.i2s`,
                html: `<p>Instruction i2s: Convert int to short</p><p>Format: i2s</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.i2s.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, truncated to a <code class="literal">short</code>, then sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert int to short`,
            };
        case 'IADD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iadd`,
                html: `<p>Instruction iadd: Add int</p><p>Format: iadd</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.iadd.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> + <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Add int`,
            };
        case 'IALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iaload`,
                html: `<p>Instruction iaload: Load int from array </p><p>Format: iaload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iaload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">int</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>value</em></span> in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved and pushed onto the operand stack. </p>`,
                tooltip: `Load int from array `,
            };
        case 'IAND':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iand`,
                html: `<p>Instruction iand: Boolean AND int</p><p>Format: iand</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.iand.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. They are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise AND (conjunction) of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean AND int`,
            };
        case 'IASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iastore`,
                html: `<p>Instruction iastore: Store into int array </p><p>Format: iastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.iastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">int</code>. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>value</em></span> is stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into int array `,
            };
        case 'ICONST_M1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_m1: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_0: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_1: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_2: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_3: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_4':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_4: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'ICONST_5':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iconst_i`,
                html: `<p>Instruction iconst_5: Push int constant </p><p>Format: iconst_[i]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>i</em></span>&gt; </p><p><a name="jvms-6.5.iconst_i.desc-100"></a> Push the <code class="literal">int</code> constant &lt;<span class="emphasis"><em>i</em></span>&gt; (-1, 0, 1, 2, 3, 4 or 5) onto the operand stack. </p>`,
                tooltip: `Push int constant `,
            };
        case 'IDIV':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.idiv`,
                html: `<p>Instruction idiv: Divide int</p><p>Format: idiv</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.idiv.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is the value of the Java programming language expression <span class="emphasis"><em>value1</em></span> / <span class="emphasis"><em>value2</em></span> (JLS §15.17.2). The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Divide int`,
            };
        case 'IF_ACMPEQ':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_acmp_cond`,
                html: `<p>Instruction if_acmpeq: Branch if reference comparison succeeds </p><p>Format: if_acmp[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_acmp_cond.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">reference</code>. They are both popped from the operand stack and compared. The results of the comparison are as follows: </p>`,
                tooltip: `Branch if reference comparison succeeds `,
            };
        case 'IF_ACMPNE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_acmp_cond`,
                html: `<p>Instruction if_acmpne: Branch if reference comparison succeeds </p><p>Format: if_acmp[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_acmp_cond.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">reference</code>. They are both popped from the operand stack and compared. The results of the comparison are as follows: </p>`,
                tooltip: `Branch if reference comparison succeeds `,
            };
        case 'IF_ICMPEQ':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_icmp_cond`,
                html: `<p>Instruction if_icmpeq: Branch if int comparison succeeds </p><p>Format: if_icmp[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_icmp_cond.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. They are both popped from the operand stack and compared. All comparisons are signed. The results of the comparison are as follows: </p>`,
                tooltip: `Branch if int comparison succeeds `,
            };
        case 'IF_ICMPNE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_icmp_cond`,
                html: `<p>Instruction if_icmpne: Branch if int comparison succeeds </p><p>Format: if_icmp[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_icmp_cond.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. They are both popped from the operand stack and compared. All comparisons are signed. The results of the comparison are as follows: </p>`,
                tooltip: `Branch if int comparison succeeds `,
            };
        case 'IFEQ':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_cond`,
                html: `<p>Instruction ifeq: Branch if int comparison with zero succeeds </p><p>Format: if[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_cond.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. It is popped from the operand stack and compared against zero. All comparisons are signed. The results of the comparisons are as follows: </p>`,
                tooltip: `Branch if int comparison with zero succeeds `,
            };
        case 'IFNE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.if_cond`,
                html: `<p>Instruction ifne: Branch if int comparison with zero succeeds </p><p>Format: if[cond] branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.if_cond.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. It is popped from the operand stack and compared against zero. All comparisons are signed. The results of the comparisons are as follows: </p>`,
                tooltip: `Branch if int comparison with zero succeeds `,
            };
        case 'IFNONNULL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ifnonnull`,
                html: `<p>Instruction ifnonnull: Branch if reference not null</p><p>Format: ifnonnull branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.ifnonnull.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">reference</code>. It is popped from the operand stack. If <span class="emphasis"><em>value</em></span> is not <code class="literal">null</code>, the unsigned <span class="emphasis"><em>branchbyte1</em></span> and <span class="emphasis"><em>branchbyte2</em></span> are used to construct a signed 16-bit offset, where the offset is calculated to be (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte2</em></span>. Execution then proceeds at that offset from the address of the opcode of this <span class="emphasis"><em>ifnonnull</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>ifnonnull</em></span> instruction. </p>`,
                tooltip: `Branch if reference not null`,
            };
        case 'IFNULL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ifnull`,
                html: `<p>Instruction ifnull: Branch if reference is null</p><p>Format: ifnull branchbyte1 branchbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.ifnull.desc-100"></a> The <span class="emphasis"><em>value</em></span> must of type <code class="literal">reference</code>. It is popped from the operand stack. If <span class="emphasis"><em>value</em></span> is <code class="literal">null</code>, the unsigned <span class="emphasis"><em>branchbyte1</em></span> and <span class="emphasis"><em>branchbyte2</em></span> are used to construct a signed 16-bit offset, where the offset is calculated to be (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte2</em></span>. Execution then proceeds at that offset from the address of the opcode of this <span class="emphasis"><em>ifnull</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>ifnull</em></span> instruction. </p>`,
                tooltip: `Branch if reference is null`,
            };
        case 'IINC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iinc`,
                html: `<p>Instruction iinc: Increment local variable by constant</p><p>Format: iinc index const</p><p>Operand Stack: No change undefined</p><p><a name="jvms-6.5.iinc.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>const</em></span> is an immediate signed byte. The local variable at <span class="emphasis"><em>index</em></span> must contain an <code class="literal">int</code>. The value <span class="emphasis"><em>const</em></span> is first sign-extended to an <code class="literal">int</code>, and then the local variable at <span class="emphasis"><em>index</em></span> is incremented by that amount. </p>`,
                tooltip: `Increment local variable by constant`,
            };
        case 'ILOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iload`,
                html: `<p>Instruction iload: Load int from local variable </p><p>Format: iload index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iload.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at <span class="emphasis"><em>index</em></span> must contain an <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> of the local variable at <span class="emphasis"><em>index</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load int from local variable `,
            };
        case 'ILOAD_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iload_n`,
                html: `<p>Instruction iload_0: Load int from local variable </p><p>Format: iload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain an <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load int from local variable `,
            };
        case 'ILOAD_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iload_n`,
                html: `<p>Instruction iload_1: Load int from local variable </p><p>Format: iload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain an <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load int from local variable `,
            };
        case 'ILOAD_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iload_n`,
                html: `<p>Instruction iload_2: Load int from local variable </p><p>Format: iload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain an <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load int from local variable `,
            };
        case 'ILOAD_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iload_n`,
                html: `<p>Instruction iload_3: Load int from local variable </p><p>Format: iload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.iload_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain an <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load int from local variable `,
            };
        case 'IMUL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.imul`,
                html: `<p>Instruction imul: Multiply int</p><p>Format: imul</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.imul.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> * <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Multiply int`,
            };
        case 'INEG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ineg`,
                html: `<p>Instruction ineg: Negate int</p><p>Format: ineg</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ineg.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. It is popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is the arithmetic negation of <span class="emphasis"><em>value</em></span>, -<span class="emphasis"><em>value</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Negate int`,
            };
        case 'INSTANCEOF':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.instanceof`,
                html: `<p>Instruction instanceof: Determine if object is of given type</p><p>Format: instanceof indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.instanceof.desc-100"></a> The <span class="emphasis"><em>objectref</em></span>, which must be of type <code class="literal">reference</code>, is popped from the operand stack. The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a class, array, or interface type. </p>`,
                tooltip: `Determine if object is of given type`,
            };
        case 'INVOKEDYNAMIC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.invokedynamic`,
                html: `<p>Instruction invokedynamic: Invoke a dynamically-computed call site</p><p>Format: invokedynamic indexbyte1 indexbyte2 0 0</p><p>Operand Stack: ..., [<span class="emphasis"><em>arg1</em></span>, [<span class="emphasis"><em>arg2</em></span> ...]] <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.invokedynamic.desc-100"></a> First, the unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a dynamically-computed call site (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>). The values of the third and fourth operand bytes must always be zero. </p>`,
                tooltip: `Invoke a dynamically-computed call site`,
            };
        case 'INVOKEINTERFACE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.invokeinterface`,
                html: `<p>Instruction invokeinterface: Invoke interface method</p><p>Format: invokeinterface indexbyte1 indexbyte2 count 0</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span>, [<span class="emphasis"><em>arg1</em></span>, [<span class="emphasis"><em>arg2</em></span> ...]] <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.invokeinterface.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to an interface method (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-4.html#jvms-4.3.3" title="4.3.3.&nbsp;Method Descriptors">§4.3.3</a>) of the interface method as well as a symbolic reference to the interface in which the interface method is to be found. The named interface method is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.4" title="5.4.3.4.&nbsp;Interface Method Resolution">§5.4.3.4</a>). </p>`,
                tooltip: `Invoke interface method`,
            };
        case 'INVOKESPECIAL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.invokespecial`,
                html: `<p>Instruction invokespecial: Invoke instance method; direct invocation of instance initialization methods and methods of the current class and its supertypes </p><p>Format: invokespecial indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span>, [<span class="emphasis"><em>arg1</em></span>, [<span class="emphasis"><em>arg2</em></span> ...]] <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.invokespecial.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a method or an interface method (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-4.html#jvms-4.3.3" title="4.3.3.&nbsp;Method Descriptors">§4.3.3</a>) of the method or interface method as well as a symbolic reference to the class or interface in which the method or interface method is to be found. The named method is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.3" title="5.4.3.3.&nbsp;Method Resolution">§5.4.3.3</a>, <a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.4" title="5.4.3.4.&nbsp;Interface Method Resolution">§5.4.3.4</a>). </p>`,
                tooltip: `Invoke instance method; direct invocation of instance initialization methods and methods of the current class and its supertypes `,
            };
        case 'INVOKESTATIC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.invokestatic`,
                html: `<p>Instruction invokestatic: Invoke a class (static) method </p><p>Format: invokestatic indexbyte1 indexbyte2</p><p>Operand Stack: ..., [<span class="emphasis"><em>arg1</em></span>, [<span class="emphasis"><em>arg2</em></span> ...]] <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.invokestatic.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a method or an interface method (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-4.html#jvms-4.3.3" title="4.3.3.&nbsp;Method Descriptors">§4.3.3</a>) of the method or interface method as well as a symbolic reference to the class or interface in which the method or interface method is to be found. The named method is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.3" title="5.4.3.3.&nbsp;Method Resolution">§5.4.3.3</a>, <a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.4" title="5.4.3.4.&nbsp;Interface Method Resolution">§5.4.3.4</a>). </p>`,
                tooltip: `Invoke a class (static) method `,
            };
        case 'INVOKEVIRTUAL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.invokevirtual`,
                html: `<p>Instruction invokevirtual: Invoke instance method; dispatch based on class</p><p>Format: invokevirtual indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span>, [<span class="emphasis"><em>arg1</em></span>, [<span class="emphasis"><em>arg2</em></span> ...]] <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.invokevirtual.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a method (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-4.html#jvms-4.3.3" title="4.3.3.&nbsp;Method Descriptors">§4.3.3</a>) of the method as well as a symbolic reference to the class in which the method is to be found. The named method is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.3" title="5.4.3.3.&nbsp;Method Resolution">§5.4.3.3</a>). </p>`,
                tooltip: `Invoke instance method; dispatch based on class`,
            };
        case 'IOR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ior`,
                html: `<p>Instruction ior: Boolean OR int</p><p>Format: ior</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ior.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. They are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise inclusive OR of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean OR int`,
            };
        case 'IREM':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.irem`,
                html: `<p>Instruction irem: Remainder int</p><p>Format: irem</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.irem.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> - (<span class="emphasis"><em>value1</em></span> / <span class="emphasis"><em>value2</em></span>) * <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Remainder int`,
            };
        case 'IRETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ireturn`,
                html: `<p>Instruction ireturn: Return int from method </p><p>Format: ireturn</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.ireturn.desc-100"></a> The current method must have return type <code class="literal">boolean</code>, <code class="literal">byte</code>, <code class="literal">char</code>, <code class="literal">short</code>, or <code class="literal">int</code>. The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, <span class="emphasis"><em>value</em></span> is popped from the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) and pushed onto the operand stack of the frame of the invoker. Any other values on the operand stack of the current method are discarded. </p>`,
                tooltip: `Return int from method `,
            };
        case 'ISHL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ishl`,
                html: `<p>Instruction ishl: Shift left int</p><p>Format: ishl</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ishl.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> left by <span class="emphasis"><em>s</em></span> bit positions, where <span class="emphasis"><em>s</em></span> is the value of the low 5 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Shift left int`,
            };
        case 'ISHR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ishr`,
                html: `<p>Instruction ishr: Arithmetic shift right int</p><p>Format: ishr</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ishr.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> right by <span class="emphasis"><em>s</em></span> bit positions, with sign extension, where <span class="emphasis"><em>s</em></span> is the value of the low 5 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Arithmetic shift right int`,
            };
        case 'ISTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.istore`,
                html: `<p>Instruction istore: Store int into local variable </p><p>Format: istore index</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.istore.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, and the value of the local variable at <span class="emphasis"><em>index</em></span> is set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store int into local variable `,
            };
        case 'ISTORE_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.istore_n`,
                html: `<p>Instruction istore_0: Store int into local variable </p><p>Format: istore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.istore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store int into local variable `,
            };
        case 'ISTORE_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.istore_n`,
                html: `<p>Instruction istore_1: Store int into local variable </p><p>Format: istore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.istore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store int into local variable `,
            };
        case 'ISTORE_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.istore_n`,
                html: `<p>Instruction istore_2: Store int into local variable </p><p>Format: istore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.istore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store int into local variable `,
            };
        case 'ISTORE_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.istore_n`,
                html: `<p>Instruction istore_3: Store int into local variable </p><p>Format: istore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.istore_n.desc-100"></a> The &lt;<span class="emphasis"><em>n</em></span>&gt; must be an index into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">int</code>. It is popped from the operand stack, and the value of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store int into local variable `,
            };
        case 'ISUB':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.isub`,
                html: `<p>Instruction isub: Subtract int</p><p>Format: isub</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.isub.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> - <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Subtract int`,
            };
        case 'IUSHR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.iushr`,
                html: `<p>Instruction iushr: Logical shift right int</p><p>Format: iushr</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.iushr.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> right by <span class="emphasis"><em>s</em></span> bit positions, with zero extension, where <span class="emphasis"><em>s</em></span> is the value of the low 5 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Logical shift right int`,
            };
        case 'IXOR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ixor`,
                html: `<p>Instruction ixor: Boolean XOR int</p><p>Format: ixor</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ixor.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. They are popped from the operand stack. An <code class="literal">int</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise exclusive OR of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean XOR int`,
            };
        case 'JSR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.jsr`,
                html: `<p>Instruction jsr: Jump subroutine</p><p>Format: jsr branchbyte1 branchbyte2</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>address</em></span></p><p><a name="jvms-6.5.jsr.desc-100"></a> The <span class="emphasis"><em>address</em></span> of the opcode of the instruction immediately following this <span class="emphasis"><em>jsr</em></span> instruction is pushed onto the operand stack as a value of type <code class="literal">returnAddress</code>. The unsigned <span class="emphasis"><em>branchbyte1</em></span> and <span class="emphasis"><em>branchbyte2</em></span> are used to construct a signed 16-bit offset, where the offset is (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte2</em></span>. Execution proceeds at that offset from the address of this <span class="emphasis"><em>jsr</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>jsr</em></span> instruction. </p>`,
                tooltip: `Jump subroutine`,
            };
        case 'JSR_W':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.jsr_w`,
                html: `<p>Instruction jsr_w: Jump subroutine (wide index)</p><p>Format: jsr_w branchbyte1 branchbyte2 branchbyte3 branchbyte4</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>address</em></span></p><p><a name="jvms-6.5.jsr_w.desc-100"></a> The <span class="emphasis"><em>address</em></span> of the opcode of the instruction immediately following this <span class="emphasis"><em>jsr_w</em></span> instruction is pushed onto the operand stack as a value of type <code class="literal">returnAddress</code>. The unsigned <span class="emphasis"><em>branchbyte1</em></span>, <span class="emphasis"><em>branchbyte2</em></span>, <span class="emphasis"><em>branchbyte3</em></span>, and <span class="emphasis"><em>branchbyte4</em></span> are used to construct a signed 32-bit offset, where the offset is (<span class="emphasis"><em>branchbyte1</em></span> <code class="literal">&lt;&lt;</code> 24) | (<span class="emphasis"><em>branchbyte2</em></span> <code class="literal">&lt;&lt;</code> 16) | (<span class="emphasis"><em>branchbyte3</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>branchbyte4</em></span>. Execution proceeds at that offset from the address of this <span class="emphasis"><em>jsr_w</em></span> instruction. The target address must be that of an opcode of an instruction within the method that contains this <span class="emphasis"><em>jsr_w</em></span> instruction. </p>`,
                tooltip: `Jump subroutine (wide index)`,
            };
        case 'L2D':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.l2d`,
                html: `<p>Instruction l2d: Convert long to double</p><p>Format: l2d</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.l2d.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack and converted to a <code class="literal">double</code> <span class="emphasis"><em>result</em></span> using the round to nearest rounding policy (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8" title="2.8.&nbsp;Floating-Point Arithmetic">§2.8</a>). The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert long to double`,
            };
        case 'L2F':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.l2f`,
                html: `<p>Instruction l2f: Convert long to float</p><p>Format: l2f</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.l2f.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack and converted to a <code class="literal">float</code> <span class="emphasis"><em>result</em></span> using the round to nearest rounding policy (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.8" title="2.8.&nbsp;Floating-Point Arithmetic">§2.8</a>). The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert long to float`,
            };
        case 'L2I':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.l2i`,
                html: `<p>Instruction l2i: Convert long to int</p><p>Format: l2i</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.l2i.desc-100"></a> The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack and converted to an <code class="literal">int</code> <span class="emphasis"><em>result</em></span> by taking the low-order 32 bits of the <code class="literal">long</code> value and discarding the high-order 32 bits. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Convert long to int`,
            };
        case 'LADD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ladd`,
                html: `<p>Instruction ladd: Add long</p><p>Format: ladd</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ladd.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. The values are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> + <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Add long`,
            };
        case 'LALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.laload`,
                html: `<p>Instruction laload: Load long from array </p><p>Format: laload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.laload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">long</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>value</em></span> in the component of the array at <span class="emphasis"><em>index</em></span> is retrieved and pushed onto the operand stack. </p>`,
                tooltip: `Load long from array `,
            };
        case 'LAND':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.land`,
                html: `<p>Instruction land: Boolean AND long</p><p>Format: land</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.land.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. They are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise AND of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean AND long`,
            };
        case 'LASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lastore`,
                html: `<p>Instruction lastore: Store into long array </p><p>Format: lastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">long</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>, and <span class="emphasis"><em>value</em></span> must be of type <code class="literal">long</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>value</em></span> is stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into long array `,
            };
        case 'LCMP':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lcmp`,
                html: `<p>Instruction lcmp: Compare long</p><p>Format: lcmp</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lcmp.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. They are both popped from the operand stack, and a signed integer comparison is performed. If <span class="emphasis"><em>value1</em></span> is greater than <span class="emphasis"><em>value2</em></span>, the <code class="literal">int</code> value 1 is pushed onto the operand stack. If <span class="emphasis"><em>value1</em></span> is equal to <span class="emphasis"><em>value2</em></span>, the <code class="literal">int</code> value 0 is pushed onto the operand stack. If <span class="emphasis"><em>value1</em></span> is less than <span class="emphasis"><em>value2</em></span>, the <code class="literal">int</code> value -1 is pushed onto the operand stack. </p>`,
                tooltip: `Compare long`,
            };
        case 'LCONST_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lconst_l`,
                html: `<p>Instruction lconst_0: Push long constant </p><p>Format: lconst_[l]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>l</em></span>&gt; </p><p><a name="jvms-6.5.lconst_l.desc-100"></a> Push the <code class="literal">long</code> constant &lt;<span class="emphasis"><em>l</em></span>&gt; (0 or 1) onto the operand stack. </p>`,
                tooltip: `Push long constant `,
            };
        case 'LCONST_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lconst_l`,
                html: `<p>Instruction lconst_1: Push long constant </p><p>Format: lconst_[l]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., &lt;<span class="emphasis"><em>l</em></span>&gt; </p><p><a name="jvms-6.5.lconst_l.desc-100"></a> Push the <code class="literal">long</code> constant &lt;<span class="emphasis"><em>l</em></span>&gt; (0 or 1) onto the operand stack. </p>`,
                tooltip: `Push long constant `,
            };
        case 'LDC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ldc`,
                html: `<p>Instruction ldc: Push item from run-time constant pool</p><p>Format: ldc index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.ldc.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte that must be a valid index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.5.5" title="2.5.5.&nbsp;Run-Time Constant Pool">§2.5.5</a>). The run-time constant pool entry at <span class="emphasis"><em>index</em></span> must be loadable (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), and not any of the following: </p>`,
                tooltip: `Push item from run-time constant pool`,
            };
        case 'LDC_W':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ldc_w`,
                html: `<p>Instruction ldc_w: Push item from run-time constant pool (wide index)</p><p>Format: ldc_w indexbyte1 indexbyte2</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.ldc_w.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are assembled into an unsigned 16-bit index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.5.5" title="2.5.5.&nbsp;Run-Time Constant Pool">§2.5.5</a>), where the value of the index is calculated as (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The index must be a valid index into the run-time constant pool of the current class. The run-time constant pool entry at the index must be loadable (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), and not any of the following: </p>`,
                tooltip: `Push item from run-time constant pool (wide index)`,
            };
        case 'LDC2_W':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ldc2_w`,
                html: `<p>Instruction ldc2_w: Push long or double from run-time constant pool (wide index) </p><p>Format: ldc2_w indexbyte1 indexbyte2</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.ldc2_w.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are assembled into an unsigned 16-bit index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.5.5" title="2.5.5.&nbsp;Run-Time Constant Pool">§2.5.5</a>), where the value of the index is calculated as (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The index must be a valid index into the run-time constant pool of the current class. The run-time constant pool entry at the index must be loadable (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), and in particular one of the following: </p>`,
                tooltip: `Push long or double from run-time constant pool (wide index) `,
            };
        case 'LDIV':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ldiv`,
                html: `<p>Instruction ldiv: Divide long</p><p>Format: ldiv</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.ldiv.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. The values are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is the value of the Java programming language expression <span class="emphasis"><em>value1</em></span> / <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Divide long`,
            };
        case 'LLOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lload`,
                html: `<p>Instruction lload: Load long from local variable </p><p>Format: lload index</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.lload.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at <span class="emphasis"><em>index</em></span> must contain a <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> of the local variable at <span class="emphasis"><em>index</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load long from local variable `,
            };
        case 'LLOAD_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lload_n`,
                html: `<p>Instruction lload_0: Load long from local variable </p><p>Format: lload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.lload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load long from local variable `,
            };
        case 'LLOAD_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lload_n`,
                html: `<p>Instruction lload_1: Load long from local variable </p><p>Format: lload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.lload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load long from local variable `,
            };
        case 'LLOAD_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lload_n`,
                html: `<p>Instruction lload_2: Load long from local variable </p><p>Format: lload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.lload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load long from local variable `,
            };
        case 'LLOAD_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lload_n`,
                html: `<p>Instruction lload_3: Load long from local variable </p><p>Format: lload_[n]</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.lload_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; must contain a <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> of the local variable at &lt;<span class="emphasis"><em>n</em></span>&gt; is pushed onto the operand stack. </p>`,
                tooltip: `Load long from local variable `,
            };
        case 'LMUL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lmul`,
                html: `<p>Instruction lmul: Multiply long</p><p>Format: lmul</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lmul.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. The values are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> * <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Multiply long`,
            };
        case 'LNEG':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lneg`,
                html: `<p>Instruction lneg: Negate long</p><p>Format: lneg</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lneg.desc-100"></a> The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">long</code>. It is popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is the arithmetic negation of <span class="emphasis"><em>value</em></span>, -<span class="emphasis"><em>value</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Negate long`,
            };
        case 'LOOKUPSWITCH':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lookupswitch`,
                html: `<p>Instruction lookupswitch: Access jump table by key match and jump</p><p>Format: lookupswitch <0-3 byte pad> defaultbyte1 defaultbyte2 defaultbyte3 defaultbyte4 npairs1 npairs2 npairs3 npairs4 match-offset pairs...</p><p>Operand Stack: ..., <span class="emphasis"><em>key</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lookupswitch.desc-100"></a> A <span class="emphasis"><em>lookupswitch</em></span> is a variable-length instruction. Immediately after the <span class="emphasis"><em>lookupswitch</em></span> opcode, between zero and three bytes must act as padding, such that <span class="emphasis"><em>defaultbyte1</em></span> begins at an address that is a multiple of four bytes from the start of the current method (the opcode of its first instruction). Immediately after the padding follow a series of signed 32-bit values: <span class="emphasis"><em>default</em></span>, <span class="emphasis"><em>npairs</em></span>, and then <span class="emphasis"><em>npairs</em></span> pairs of signed 32-bit values. The <span class="emphasis"><em>npairs</em></span> must be greater than or equal to 0. Each of the <span class="emphasis"><em>npairs</em></span> pairs consists of an <code class="literal">int</code> <span class="emphasis"><em>match</em></span> and a signed 32-bit <span class="emphasis"><em>offset</em></span>. Each of these signed 32-bit values is constructed from four unsigned bytes as (<span class="emphasis"><em>byte1</em></span> <code class="literal">&lt;&lt;</code> 24) | (<span class="emphasis"><em>byte2</em></span> <code class="literal">&lt;&lt;</code> 16) | (<span class="emphasis"><em>byte3</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>byte4</em></span>. </p>`,
                tooltip: `Access jump table by key match and jump`,
            };
        case 'LOR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lor`,
                html: `<p>Instruction lor: Boolean OR long</p><p>Format: lor</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lor.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. They are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise inclusive OR of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean OR long`,
            };
        case 'LREM':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lrem`,
                html: `<p>Instruction lrem: Remainder long</p><p>Format: lrem</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lrem.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. The values are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> - (<span class="emphasis"><em>value1</em></span> / <span class="emphasis"><em>value2</em></span>) * <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Remainder long`,
            };
        case 'LRETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lreturn`,
                html: `<p>Instruction lreturn: Return long from method </p><p>Format: lreturn</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.lreturn.desc-100"></a> The current method must have return type <code class="literal">long</code>. The <span class="emphasis"><em>value</em></span> must be of type <code class="literal">long</code>. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, <span class="emphasis"><em>value</em></span> is popped from the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) and pushed onto the operand stack of the frame of the invoker. Any other values on the operand stack of the current method are discarded. </p>`,
                tooltip: `Return long from method `,
            };
        case 'LSHL':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lshl`,
                html: `<p>Instruction lshl: Shift left long</p><p>Format: lshl</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lshl.desc-100"></a> The <span class="emphasis"><em>value1</em></span> must be of type <code class="literal">long</code>, and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> left by <span class="emphasis"><em>s</em></span> bit positions, where <span class="emphasis"><em>s</em></span> is the low 6 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Shift left long`,
            };
        case 'LSHR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lshr`,
                html: `<p>Instruction lshr: Arithmetic shift right long</p><p>Format: lshr</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lshr.desc-100"></a> The <span class="emphasis"><em>value1</em></span> must be of type <code class="literal">long</code>, and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> right by <span class="emphasis"><em>s</em></span> bit positions, with sign extension, where <span class="emphasis"><em>s</em></span> is the value of the low 6 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Arithmetic shift right long`,
            };
        case 'LSTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lstore`,
                html: `<p>Instruction lstore: Store long into local variable </p><p>Format: lstore index</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lstore.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack, and the local variables at <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>index</em></span>+1 are set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store long into local variable `,
            };
        case 'LSTORE_0':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lstore_n`,
                html: `<p>Instruction lstore_0: Store long into local variable </p><p>Format: lstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack, and the local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store long into local variable `,
            };
        case 'LSTORE_1':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lstore_n`,
                html: `<p>Instruction lstore_1: Store long into local variable </p><p>Format: lstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack, and the local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store long into local variable `,
            };
        case 'LSTORE_2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lstore_n`,
                html: `<p>Instruction lstore_2: Store long into local variable </p><p>Format: lstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack, and the local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store long into local variable `,
            };
        case 'LSTORE_3':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lstore_n`,
                html: `<p>Instruction lstore_3: Store long into local variable </p><p>Format: lstore_[n]</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.lstore_n.desc-100"></a> Both &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 must be indices into the local variable array of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>). The <span class="emphasis"><em>value</em></span> on the top of the operand stack must be of type <code class="literal">long</code>. It is popped from the operand stack, and the local variables at &lt;<span class="emphasis"><em>n</em></span>&gt; and &lt;<span class="emphasis"><em>n</em></span>&gt;+1 are set to <span class="emphasis"><em>value</em></span>. </p>`,
                tooltip: `Store long into local variable `,
            };
        case 'LSUB':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lsub`,
                html: `<p>Instruction lsub: Subtract long</p><p>Format: lsub</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lsub.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. The values are popped from the operand stack. The <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is <span class="emphasis"><em>value1</em></span> - <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Subtract long`,
            };
        case 'LUSHR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lushr`,
                html: `<p>Instruction lushr: Logical shift right long</p><p>Format: lushr</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lushr.desc-100"></a> The <span class="emphasis"><em>value1</em></span> must be of type <code class="literal">long</code>, and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">int</code>. The values are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by shifting <span class="emphasis"><em>value1</em></span> right logically by <span class="emphasis"><em>s</em></span> bit positions, with zero extension, where <span class="emphasis"><em>s</em></span> is the value of the low 6 bits of <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Logical shift right long`,
            };
        case 'LXOR':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.lxor`,
                html: `<p>Instruction lxor: Boolean XOR long</p><p>Format: lxor</p><p>Operand Stack: ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>result</em></span></p><p><a name="jvms-6.5.lxor.desc-100"></a> Both <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span> must be of type <code class="literal">long</code>. They are popped from the operand stack. A <code class="literal">long</code> <span class="emphasis"><em>result</em></span> is calculated by taking the bitwise exclusive OR of <span class="emphasis"><em>value1</em></span> and <span class="emphasis"><em>value2</em></span>. The <span class="emphasis"><em>result</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Boolean XOR long`,
            };
        case 'MONITORENTER':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorenter`,
                html: `<p>Instruction monitorenter: Enter monitor for object</p><p>Format: monitorenter</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ... </p><p><a name="jvms-6.5.monitorenter.desc-100"></a> The <span class="emphasis"><em>objectref</em></span> must be of type <code class="literal">reference</code>. </p>`,
                tooltip: `Enter monitor for object`,
            };
        case 'MONITOREXIT':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit`,
                html: `<p>Instruction monitorexit: Exit monitor for object</p><p>Format: monitorexit</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span> <span class="symbol">→</span> ... </p><p><a name="jvms-6.5.monitorexit.desc-100"></a> The <span class="emphasis"><em>objectref</em></span> must be of type <code class="literal">reference</code>. </p>`,
                tooltip: `Exit monitor for object`,
            };
        case 'MULTIANEWARRAY':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.multianewarray`,
                html: `<p>Instruction multianewarray: Create new multidimensional array</p><p>Format: multianewarray indexbyte1 indexbyte2 dimensions</p><p>Operand Stack: ..., <span class="emphasis"><em>count1</em></span>, [<span class="emphasis"><em>count2</em></span>, ...] <span class="symbol">→</span> ..., <span class="emphasis"><em>arrayref</em></span></p><p><a name="jvms-6.5.multianewarray.desc-100"></a> The <span class="emphasis"><em>dimensions</em></span> operand is an unsigned byte that must be greater than or equal to 1. It represents the number of dimensions of the array to be created. The operand stack must contain <span class="emphasis"><em>dimensions</em></span> values. Each such value represents the number of components in a dimension of the array to be created, must be of type <code class="literal">int</code>, and must be non-negative. The <span class="emphasis"><em>count1</em></span> is the desired length in the first dimension, <span class="emphasis"><em>count2</em></span> in the second, etc. </p>`,
                tooltip: `Create new multidimensional array`,
            };
        case 'NEW':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.new`,
                html: `<p>Instruction new: Create new object</p><p>Format: new indexbyte1 indexbyte2</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>objectref</em></span></p><p><a name="jvms-6.5.new.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a class or interface type. The named class or interface type is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.1" title="5.4.3.1.&nbsp;Class and Interface Resolution">§5.4.3.1</a>) and should result in a class type. Memory for a new instance of that class is allocated from the garbage-collected heap, and the instance variables of the new object are initialized to their default initial values (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.3" title="2.3.&nbsp;Primitive Types and Values">§2.3</a>, <a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.4" title="2.4.&nbsp;Reference Types and Values">§2.4</a>). The <span class="emphasis"><em>objectref</em></span>, a <code class="literal">reference</code> to the instance, is pushed onto the operand stack. </p>`,
                tooltip: `Create new object`,
            };
        case 'NEWARRAY':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.newarray`,
                html: `<p>Instruction newarray: Create new array</p><p>Format: newarray atype</p><p>Operand Stack: ..., <span class="emphasis"><em>count</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>arrayref</em></span></p><p><a name="jvms-6.5.newarray.desc-100"></a> The <span class="emphasis"><em>count</em></span> must be of type <code class="literal">int</code>. It is popped off the operand stack. The <span class="emphasis"><em>count</em></span> represents the number of elements in the array to be created. </p>`,
                tooltip: `Create new array`,
            };
        case 'NOP':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.nop`,
                html: `<p>Instruction nop: Do nothing</p><p>Format: nop</p><p>Operand Stack: No change undefined</p><p><a name="jvms-6.5.nop.desc-100"></a> Do nothing. </p>`,
                tooltip: `Do nothing`,
            };
        case 'POP':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.pop`,
                html: `<p>Instruction pop: Pop the top operand stack value</p><p>Format: pop</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.pop.desc-100"></a> Pop the top value from the operand stack. </p>`,
                tooltip: `Pop the top operand stack value`,
            };
        case 'POP2':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.pop2`,
                html: `<p>Instruction pop2: Pop the top one or two operand stack values</p><p>Format: pop2</p><p>Operand Stack: <a name="jvms-6.5.pop2.stack-100"></a>Form 1: <a name="jvms-6.5.pop2.stack-100-A"></a>..., <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span></p><p><a name="jvms-6.5.pop2.desc-100"></a> Pop the top one or two values from the operand stack. </p>`,
                tooltip: `Pop the top one or two operand stack values`,
            };
        case 'PUTFIELD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.putfield`,
                html: `<p>Instruction putfield: Set field in object</p><p>Format: putfield indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>objectref</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.putfield.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a field (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor of the field as well as a symbolic reference to the class in which the field is to be found. The referenced field is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.2" title="5.4.3.2.&nbsp;Field Resolution">§5.4.3.2</a>). </p>`,
                tooltip: `Set field in object`,
            };
        case 'PUTSTATIC':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.putstatic`,
                html: `<p>Instruction putstatic: Set static field in class</p><p>Format: putstatic indexbyte1 indexbyte2</p><p>Operand Stack: ..., <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.putstatic.desc-100"></a> The unsigned <span class="emphasis"><em>indexbyte1</em></span> and <span class="emphasis"><em>indexbyte2</em></span> are used to construct an index into the run-time constant pool of the current class (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>), where the value of the index is (<span class="emphasis"><em>indexbyte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>indexbyte2</em></span>. The run-time constant pool entry at the index must be a symbolic reference to a field (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.1" title="5.1.&nbsp;The Run-Time Constant Pool">§5.1</a>), which gives the name and descriptor of the field as well as a symbolic reference to the class or interface in which the field is to be found. The referenced field is resolved (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-5.html#jvms-5.4.3.2" title="5.4.3.2.&nbsp;Field Resolution">§5.4.3.2</a>). </p>`,
                tooltip: `Set static field in class`,
            };
        case 'RET':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.ret`,
                html: `<p>Instruction ret: Return from subroutine</p><p>Format: ret index</p><p>Operand Stack: No change undefined</p><p><a name="jvms-6.5.ret.desc-100"></a> The <span class="emphasis"><em>index</em></span> is an unsigned byte between 0 and 255, inclusive. The local variable at <span class="emphasis"><em>index</em></span> in the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) must contain a value of type <code class="literal">returnAddress</code>. The contents of the local variable are written into the Java Virtual Machine's <code class="literal">pc</code> register, and execution continues there. </p>`,
                tooltip: `Return from subroutine`,
            };
        case 'RETURN':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.return`,
                html: `<p>Instruction return: Return void from method </p><p>Format: return</p><p>Operand Stack: ... <span class="symbol">→</span> [empty]</p><p><a name="jvms-6.5.return.desc-100"></a> The current method must have return type <code class="literal">void</code>. If the current method is a <code class="literal">synchronized</code> method, the monitor entered or reentered on invocation of the method is updated and possibly exited as if by execution of a <span class="emphasis"><em>monitorexit</em></span> instruction (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.monitorexit" title="monitorexit">§<span class="emphasis"><em>monitorexit</em></span></a>) in the current thread. If no exception is thrown, any values on the operand stack of the current frame (<a class="xref" href="https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-2.html#jvms-2.6" title="2.6.&nbsp;Frames">§2.6</a>) are discarded. </p>`,
                tooltip: `Return void from method `,
            };
        case 'SALOAD':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.saload`,
                html: `<p>Instruction saload: Load short from array </p><p>Format: saload</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.saload.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">short</code>. The <span class="emphasis"><em>index</em></span> must be of type <code class="literal">int</code>. Both <span class="emphasis"><em>arrayref</em></span> and <span class="emphasis"><em>index</em></span> are popped from the operand stack. The component of the array at <span class="emphasis"><em>index</em></span> is retrieved and sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>value</em></span>. That <span class="emphasis"><em>value</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Load short from array `,
            };
        case 'SASTORE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.sastore`,
                html: `<p>Instruction sastore: Store into short array </p><p>Format: sastore</p><p>Operand Stack: ..., <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, <span class="emphasis"><em>value</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.sastore.desc-100"></a> The <span class="emphasis"><em>arrayref</em></span> must be of type <code class="literal">reference</code> and must refer to an array whose components are of type <code class="literal">short</code>. Both <span class="emphasis"><em>index</em></span> and <span class="emphasis"><em>value</em></span> must be of type <code class="literal">int</code>. The <span class="emphasis"><em>arrayref</em></span>, <span class="emphasis"><em>index</em></span>, and <span class="emphasis"><em>value</em></span> are popped from the operand stack. The <code class="literal">int</code> <span class="emphasis"><em>value</em></span> is truncated to a <code class="literal">short</code> and stored as the component of the array indexed by <span class="emphasis"><em>index</em></span>. </p>`,
                tooltip: `Store into short array `,
            };
        case 'SIPUSH':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.sipush`,
                html: `<p>Instruction sipush: Push short</p><p>Format: sipush byte1 byte2</p><p>Operand Stack: ... <span class="symbol">→</span> ..., <span class="emphasis"><em>value</em></span></p><p><a name="jvms-6.5.sipush.desc-100"></a> The immediate unsigned <span class="emphasis"><em>byte1</em></span> and <span class="emphasis"><em>byte2</em></span> values are assembled into an intermediate <code class="literal">short</code>, where the value of the <code class="literal">short</code> is (<span class="emphasis"><em>byte1</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>byte2</em></span>. The intermediate value is then sign-extended to an <code class="literal">int</code> <span class="emphasis"><em>value</em></span>. That <span class="emphasis"><em>value</em></span> is pushed onto the operand stack. </p>`,
                tooltip: `Push short`,
            };
        case 'SWAP':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.swap`,
                html: `<p>Instruction swap: Swap the top two operand stack values</p><p>Format: swap</p><p>Operand Stack: ..., <span class="emphasis"><em>value2</em></span>, <span class="emphasis"><em>value1</em></span> <span class="symbol">→</span> ..., <span class="emphasis"><em>value1</em></span>, <span class="emphasis"><em>value2</em></span></p><p><a name="jvms-6.5.swap.desc-100"></a> Swap the top two values on the operand stack. </p>`,
                tooltip: `Swap the top two operand stack values`,
            };
        case 'TABLESWITCH':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.tableswitch`,
                html: `<p>Instruction tableswitch: Access jump table by index and jump</p><p>Format: tableswitch <0-3 byte pad> defaultbyte1 defaultbyte2 defaultbyte3 defaultbyte4 lowbyte1 lowbyte2 lowbyte3 lowbyte4 highbyte1 highbyte2 highbyte3 highbyte4 jump offsets...</p><p>Operand Stack: ..., <span class="emphasis"><em>index</em></span> <span class="symbol">→</span> ...</p><p><a name="jvms-6.5.tableswitch.desc-100"></a> A <span class="emphasis"><em>tableswitch</em></span> is a variable-length instruction. Immediately after the <span class="emphasis"><em>tableswitch</em></span> opcode, between zero and three bytes must act as padding, such that <span class="emphasis"><em>defaultbyte1</em></span> begins at an address that is a multiple of four bytes from the start of the current method (the opcode of its first instruction). Immediately after the padding are bytes constituting three signed 32-bit values: <span class="emphasis"><em>default</em></span>, <span class="emphasis"><em>low</em></span>, and <span class="emphasis"><em>high</em></span>. Immediately following are bytes constituting a series of <span class="emphasis"><em>high</em></span> - <span class="emphasis"><em>low</em></span> + 1 signed 32-bit offsets. The value <span class="emphasis"><em>low</em></span> must be less than or equal to <span class="emphasis"><em>high</em></span>. The <span class="emphasis"><em>high</em></span> - <span class="emphasis"><em>low</em></span> + 1 signed 32-bit offsets are treated as a 0-based jump table. Each of these signed 32-bit values is constructed as (<span class="emphasis"><em>byte1</em></span> <code class="literal">&lt;&lt;</code> 24) | (<span class="emphasis"><em>byte2</em></span> <code class="literal">&lt;&lt;</code> 16) | (<span class="emphasis"><em>byte3</em></span> <code class="literal">&lt;&lt;</code> 8) | <span class="emphasis"><em>byte4</em></span>. </p>`,
                tooltip: `Access jump table by index and jump`,
            };
        case 'WIDE':
            return {
                url: `https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html#jvms-6.5.wide`,
                html: `<p>Instruction wide: Extend local variable index by additional bytes</p><p>Format: wide <opcode> indexbyte1 indexbyte2</p><p>Operand Stack: <span class="emphasis"><em>wide</em></span> = 196 (0xc4) undefined</p><p>null</p>`,
                tooltip: `Extend local variable index by additional bytes`,
            };
    }
}
