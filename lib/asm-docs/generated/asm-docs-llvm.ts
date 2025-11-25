import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case 'RET':
            return {
                url: `https://llvm.org/docs/LangRef.html#ret-instruction`,
                html: `<html><head></head><body><span id="i-ret"></span><h4><a class="toc-backref" href="#id2156" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ Instruction</a></h4>
<section id="id32">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">ret</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">Return</span> <span class="n">a</span> <span class="n">value</span> <span class="kn">from</span> <span class="nn">a</span> <span class="n">non</span><span class="o">-</span><span class="n">void</span> <span class="n">function</span>
<span class="n">ret</span> <span class="n">void</span>                 <span class="p">;</span> <span class="n">Return</span> <span class="kn">from</span> <span class="nn">void</span> <span class="n">function</span>
</pre></div>
</div>
</section>
<section id="overview">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction is used to return control flow (and optionally
a value) from a function back to the caller.</p>
<p>There are two forms of the ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction: one that returns a
value and then causes control flow, and one that just causes control
flow to occur.</p>
</section>
<section id="arguments">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction optionally accepts a single argument, the
return value. The type of the return value must be a ‘<a class="reference internal" href="#t-firstclass"><span class="std std-ref">first
class</span></a>’ type.</p>
<p>A function is not <a class="reference internal" href="#wellformed"><span class="std std-ref">well formed</span></a> if it has a non-void
return type and contains a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction with no return value or
a return value with a type that does not match its type, or if it has a
void return type and contains a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction with a return
value.</p>
</section>
<section id="id33">
<h5>Semantics:</h5>
<p>When the ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction is executed, control flow returns back to
the calling function’s context. If the caller is a
“<a class="reference internal" href="#i-call"><span class="std std-ref">call</span></a>” instruction, execution continues at the
instruction after the call. If the caller was an
“<a class="reference internal" href="#i-invoke"><span class="std std-ref">invoke</span></a>” instruction, execution continues at the
beginning of the “normal” destination block. If the instruction returns
a value, that value shall set the call or invoke instruction’s return
value.</p>
</section>
<section id="example">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">ret</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">5</span><span class="w">                       </span><span class="c">; Return an integer value of 5</span>
<span class="k">ret</span><span class="w"> </span><span class="k">void</span><span class="w">                        </span><span class="c">; Return from a void function</span>
<span class="k">ret</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">4</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">2</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="c">; Return a struct of values 4 and 2</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘ret’ instruction is used to return control flow (and optionallya value) from a function back to the caller.There are two forms of the ‘ret’ instruction: one that returns avalue and then causes control flow, and one that just causes controlflow to occur.`,
            };
        case 'BR':
            return {
                url: `https://llvm.org/docs/LangRef.html#br-instruction`,
                html: `<html><head></head><body><span id="i-br"></span><h4><a class="toc-backref" href="#id2157" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ Instruction</a></h4>
<section id="id34">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">br</span> <span class="n">i1</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">iftrue</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">iffalse</span><span class="o">&gt;</span>
<span class="n">br</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">Unconditional</span> <span class="n">branch</span>
</pre></div>
</div>
</section>
<section id="id35">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction is used to cause control flow to transfer to a
different basic block in the current function. There are two forms of
this instruction, corresponding to a conditional branch and an
unconditional branch.</p>
</section>
<section id="id36">
<h5>Arguments:</h5>
<p>The conditional branch form of the ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction takes a single
‘<code class="docutils literal notranslate"><span class="pre">i1</span></code>’ value and two ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ values. The unconditional form of the
‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction takes a single ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ value as a target.</p>
</section>
<section id="id37">
<h5>Semantics:</h5>
<p>Upon execution of a conditional ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction, the ‘<code class="docutils literal notranslate"><span class="pre">i1</span></code>’
argument is evaluated. If the value is <code class="docutils literal notranslate"><span class="pre">true</span></code>, control flows to the
‘<code class="docutils literal notranslate"><span class="pre">iftrue</span></code>’ <code class="docutils literal notranslate"><span class="pre">label</span></code> argument. If “cond” is <code class="docutils literal notranslate"><span class="pre">false</span></code>, control flows
to the ‘<code class="docutils literal notranslate"><span class="pre">iffalse</span></code>’ <code class="docutils literal notranslate"><span class="pre">label</span></code> argument.
If ‘<code class="docutils literal notranslate"><span class="pre">cond</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</section>
<section id="id38">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">Test:</span>
<span class="w">  </span><span class="nv">%cond</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">icmp</span><span class="w"> </span><span class="k">eq</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%a</span><span class="p">,</span><span class="w"> </span><span class="nv">%b</span>
<span class="w">  </span><span class="k">br</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="nv">%cond</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%IfEqual</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%IfUnequal</span>
<span class="nl">IfEqual:</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="nl">IfUnequal:</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘br’ instruction is used to cause control flow to transfer to adifferent basic block in the current function. There are two forms ofthis instruction, corresponding to a conditional branch and anunconditional branch.`,
            };
        case 'SWITCH':
            return {
                url: `https://llvm.org/docs/LangRef.html#switch-instruction`,
                html: `<html><head></head><body><span id="i-switch"></span><h4><a class="toc-backref" href="#id2158" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ Instruction</a></h4>
<section id="id39">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">switch</span> <span class="o">&lt;</span><span class="n">intty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">defaultdest</span><span class="o">&gt;</span> <span class="p">[</span> <span class="o">&lt;</span><span class="n">intty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest</span><span class="o">&gt;</span> <span class="o">...</span> <span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id40">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction is used to transfer control flow to one of
several different places. It is a generalization of the ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’
instruction, allowing a branch to occur to one of many possible
destinations.</p>
</section>
<section id="id41">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction uses three parameters: an integer
comparison value ‘<code class="docutils literal notranslate"><span class="pre">value</span></code>’, a default ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ destination, and an
array of pairs of comparison value constants and ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’s. The table
is not allowed to contain duplicate constant entries.</p>
</section>
<section id="id42">
<h5>Semantics:</h5>
<p>The <code class="docutils literal notranslate"><span class="pre">switch</span></code> instruction specifies a table of values and destinations.
When the ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction is executed, this table is searched
for the given value. If the value is found, control flow is transferred
to the corresponding destination; otherwise, control flow is transferred
to the default destination.
If ‘<code class="docutils literal notranslate"><span class="pre">value</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</section>
<section id="implementation">
<h5>Implementation:</h5>
<p>Depending on properties of the target machine and the particular
<code class="docutils literal notranslate"><span class="pre">switch</span></code> instruction, this instruction may be code generated in
different ways. For example, it could be generated as a series of
chained conditional branches or with a lookup table.</p>
</section>
<section id="id43">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; Emulate a conditional br instruction</span>
<span class="nv">%Val</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="nv">%value</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span>
<span class="k">switch</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%Val</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%truedest</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%falsedest</span><span class="w"> </span><span class="p">]</span>

<span class="c">; Emulate an unconditional br instruction</span>
<span class="k">switch</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%dest</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="p">]</span>

<span class="c">; Implement a jump table:</span>
<span class="k">switch</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%val</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%otherwise</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%onzero</span>
<span class="w">                                    </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%onone</span>
<span class="w">                                    </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%ontwo</span><span class="w"> </span><span class="p">]</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘switch’ instruction is used to transfer control flow to one ofseveral different places. It is a generalization of the ‘br’instruction, allowing a branch to occur to one of many possibledestinations.`,
            };
        case 'INDIRECTBR':
            return {
                url: `https://llvm.org/docs/LangRef.html#indirectbr-instruction`,
                html: `<html><head></head><body><span id="i-indirectbr"></span><h4><a class="toc-backref" href="#id2159" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">indirectbr</span></code>’ Instruction</a></h4>
<section id="id44">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">indirectbr</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">address</span><span class="o">&gt;</span><span class="p">,</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id45">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">indirectbr</span></code>’ instruction implements an indirect branch to a
label within the current function, whose address is specified by
“<code class="docutils literal notranslate"><span class="pre">address</span></code>”. Address must be derived from a
<a class="reference internal" href="#blockaddress"><span class="std std-ref">blockaddress</span></a> constant.</p>
</section>
<section id="id46">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">address</span></code>’ argument is the address of the label to jump to. The
rest of the arguments indicate the full set of possible destinations
that the address may point to. Blocks are allowed to occur multiple
times in the destination list, though this isn’t particularly useful.</p>
<p>This destination list is required so that dataflow analysis has an
accurate understanding of the CFG.</p>
</section>
<section id="id47">
<h5>Semantics:</h5>
<p>Control transfers to the block specified in the address argument. All
possible destination blocks must be listed in the label list, otherwise
this instruction has undefined behavior. This implies that jumps to
labels defined in other functions have undefined behavior as well.
If ‘<code class="docutils literal notranslate"><span class="pre">address</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</section>
<section id="id48">
<h5>Implementation:</h5>
<p>This is typically implemented with a jump through a register.</p>
</section>
<section id="id49">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">indirectbr</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%Addr</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%bb1</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%bb2</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%bb3</span><span class="w"> </span><span class="p">]</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘indirectbr’ instruction implements an indirect branch to alabel within the current function, whose address is specified by“address”. Address must be derived from ablockaddress constant.`,
            };
        case 'INVOKE':
            return {
                url: `https://llvm.org/docs/LangRef.html#invoke-instruction`,
                html: `<html><head></head><body><span id="i-invoke"></span><h4><a class="toc-backref" href="#id2160" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ Instruction</a></h4>
<section id="id50">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">invoke</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span>
              <span class="p">[</span><span class="n">operand</span> <span class="n">bundles</span><span class="p">]</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">normal</span> <span class="n">label</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">exception</span> <span class="n">label</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id51">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ instruction causes control to transfer to a specified
function, with the possibility of control flow transfer to either the
‘<code class="docutils literal notranslate"><span class="pre">normal</span></code>’ label or the ‘<code class="docutils literal notranslate"><span class="pre">exception</span></code>’ label. If the callee function
returns with the “<code class="docutils literal notranslate"><span class="pre">ret</span></code>” instruction, control flow will return to the
“normal” label. If the callee (or any indirect callees) returns via the
“<a class="reference internal" href="#i-resume"><span class="std std-ref">resume</span></a>” instruction or other exception handling
mechanism, control is interrupted and continued at the dynamically
nearest “exception” label.</p>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">exception</span></code>’ label is a <a class="reference external" href="ExceptionHandling.html#overview">landing
pad</a> for the exception. As such,
‘<code class="docutils literal notranslate"><span class="pre">exception</span></code>’ label is required to have the
“<a class="reference internal" href="#i-landingpad"><span class="std std-ref">landingpad</span></a>” instruction, which contains the
information about the behavior of the program after unwinding happens,
as its first non-PHI instruction. The restrictions on the
“<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>” instruction’s tightly couples it to the “<code class="docutils literal notranslate"><span class="pre">invoke</span></code>”
instruction, so that the important information contained within the
“<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>” instruction can’t be lost through normal code motion.</p>
</section>
<section id="id52">
<h5>Arguments:</h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic simple">
<li><p>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions.</p></li>
<li><p>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">noext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’
attributes are valid here.</p></li>
<li><p>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>. The signature is computed based on the return type and argument
types.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being invoked. The
argument types must match the types implied by this signature. This
is only required if the signature specifies a varargs type.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be invoked. In most cases, this is a direct function invocation, but
indirect <code class="docutils literal notranslate"><span class="pre">invoke</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">normal</span> <span class="pre">label</span></code>’: the label reached when the called function
executes a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">exception</span> <span class="pre">label</span></code>’: the label reached when a callee returns via
the <a class="reference internal" href="#i-resume"><span class="std std-ref">resume</span></a> instruction or other exception handling
mechanism.</p></li>
<li><p>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</p></li>
<li><p>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</p></li>
</ol>
</section>
<section id="id53">
<h5>Semantics:</h5>
<p>This instruction is designed to operate as a standard ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’
instruction in most regards. The primary difference is that it
establishes an association with a label, which is used by the runtime
library to unwind the stack.</p>
<p>This instruction is used in languages with destructors to ensure that
proper cleanup is performed in the case of either a <code class="docutils literal notranslate"><span class="pre">longjmp</span></code> or a
thrown exception. Additionally, this is important for implementation of
‘<code class="docutils literal notranslate"><span class="pre">catch</span></code>’ clauses in high-level languages that support them.</p>
<p>For the purposes of the SSA form, the definition of the value returned
by the ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ instruction is deemed to occur on the edge from the
current block to the “normal” label. If the callee unwinds then no
return value is available.</p>
</section>
<section id="id54">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%retval</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">invoke</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="vg">@Test</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="m">15</span><span class="p">)</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%Continue</span>
<span class="w">            </span><span class="k">unwind</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%TestCleanup</span><span class="w">              </span><span class="c">; i32:retval set</span>
<span class="nv">%retval</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">invoke</span><span class="w"> </span><span class="k">coldcc</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%Testfnptr</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="m">15</span><span class="p">)</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%Continue</span>
<span class="w">            </span><span class="k">unwind</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%TestCleanup</span><span class="w">              </span><span class="c">; i32:retval set</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘invoke’ instruction causes control to transfer to a specifiedfunction, with the possibility of control flow transfer to either the‘normal’ label or the ‘exception’ label. If the callee functionreturns with the “ret” instruction, control flow will return to the“normal” label. If the callee (or any indirect callees) returns via the“resume” instruction or other exception handlingmechanism, control is interrupted and continued at the dynamicallynearest “exception” label.The ‘exception’ label is a landingpad for the exception. As such,‘exception’ label is required to have the“landingpad” instruction, which contains theinformation about the behavior of the program after unwinding happens,as its first non-PHI instruction. The restrictions on the“landingpad” instruction’s tightly couples it to the “invoke”instruction, so that the important information contained within the“landingpad” instruction can’t be lost through normal code motion.`,
            };
        case 'CALLBR':
            return {
                url: `https://llvm.org/docs/LangRef.html#callbr-instruction`,
                html: `<html><head></head><body><span id="i-callbr"></span><h4><a class="toc-backref" href="#id2161" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ Instruction</a></h4>
<section id="id55">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">callbr</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span>
              <span class="p">[</span><span class="n">operand</span> <span class="n">bundles</span><span class="p">]</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">fallthrough</span> <span class="n">label</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">indirect</span> <span class="n">labels</span><span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id56">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ instruction causes control to transfer to a specified
function, with the possibility of control flow transfer to either the
‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span></code>’ label or one of the ‘<code class="docutils literal notranslate"><span class="pre">indirect</span></code>’ labels.</p>
<p>This instruction should only be used to implement the “goto” feature of gcc
style inline assembly. Any other usage is an error in the IR verifier.</p>
<p>Note that in order to support outputs along indirect edges, LLVM may need to
split critical edges, which may require synthesizing a replacement block for
the <code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code>. Therefore, the address of a label as seen by another
<code class="docutils literal notranslate"><span class="pre">callbr</span></code> instruction, or for a <a class="reference internal" href="#blockaddress"><span class="std std-ref">blockaddress</span></a> constant,
may not be equal to the address provided for the same block to this
instruction’s <code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code> operand. The assembly code may only transfer
control to addresses provided via this instruction’s <code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code>.</p>
<p>On target architectures that implement branch target enforcement by requiring
indirect (register-controlled) branch instructions to jump only to locations
marked by a special instruction (such as AArch64 <code class="docutils literal notranslate"><span class="pre">bti</span></code>), the called code is
expected not to use such an indirect branch to transfer control to the
locations in <code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code>. Therefore, including a label in the
<code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code> of a <code class="docutils literal notranslate"><span class="pre">callbr</span></code> does not require the compiler to put a
<code class="docutils literal notranslate"><span class="pre">bti</span></code> or equivalent instruction at the label.</p>
</section>
<section id="id57">
<h5>Arguments:</h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic simple">
<li><p>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions.</p></li>
<li><p>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">noext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’
attributes are valid here.</p></li>
<li><p>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>.  The signature is computed based on the return type and argument
types.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being called. The
argument types must match the types implied by this signature. This
is only required if the signature specifies a varargs type.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be called. In most cases, this is a direct function call, but
other <code class="docutils literal notranslate"><span class="pre">callbr</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span> <span class="pre">label</span></code>’: the label reached when the inline assembly’s
execution exits the bottom.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code>’: the labels reached when a callee transfers control
to a location other than the ‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span> <span class="pre">label</span></code>’. Label constraints
refer to these destinations.</p></li>
<li><p>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</p></li>
<li><p>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</p></li>
</ol>
</section>
<section id="id58">
<h5>Semantics:</h5>
<p>This instruction is designed to operate as a standard ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’
instruction in most regards. The primary difference is that it
establishes an association with additional labels to define where control
flow goes after the call.</p>
<p>The output values of a ‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ instruction are available both in the
the ‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span></code>’ block, and any ‘<code class="docutils literal notranslate"><span class="pre">indirect</span></code>’ blocks(s).</p>
<p>The only use of this today is to implement the “goto” feature of gcc inline
assembly where additional labels can be provided as locations for the inline
assembly to jump to.</p>
</section>
<section id="id59">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; "asm goto" without output constraints.</span>
<span class="k">callbr</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="k">asm</span><span class="w"> </span><span class="s">""</span><span class="p">,</span><span class="w"> </span><span class="s">"r,!i"</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span>
<span class="w">            </span><span class="k">to</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%fallthrough</span><span class="w"> </span><span class="p">[</span><span class="kt">label</span><span class="w"> </span><span class="nv">%indirect</span><span class="p">]</span>

<span class="c">; "asm goto" with output constraints.</span>
<span class="p">&lt;</span><span class="err">res</span><span class="k">ult</span><span class="p">&gt;</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">callbr</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="k">asm</span><span class="w"> </span><span class="s">""</span><span class="p">,</span><span class="w"> </span><span class="s">"=r,r,!i"</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span>
<span class="w">            </span><span class="k">to</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%fallthrough</span><span class="w"> </span><span class="p">[</span><span class="kt">label</span><span class="w"> </span><span class="nv">%indirect</span><span class="p">]</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘callbr’ instruction causes control to transfer to a specifiedfunction, with the possibility of control flow transfer to either the‘fallthrough’ label or one of the ‘indirect’ labels.This instruction should only be used to implement the “goto” feature of gccstyle inline assembly. Any other usage is an error in the IR verifier.Note that in order to support outputs along indirect edges, LLVM may need tosplit critical edges, which may require synthesizing a replacement block forthe indirect labels. Therefore, the address of a label as seen by anothercallbr instruction, or for a blockaddress constant,may not be equal to the address provided for the same block to thisinstruction’s indirect labels operand. The assembly code may only transfercontrol to addresses provided via this instruction’s indirect labels.On target architectures that implement branch target enforcement by requiringindirect (register-controlled) branch instructions to jump only to locationsmarked by a special instruction (such as AArch64 bti), the called code isexpected not to use such an indirect branch to transfer control to thelocations in indirect labels. Therefore, including a label in theindirect labels of a callbr does not require the compiler to put abti or equivalent instruction at the label.`,
            };
        case 'RESUME':
            return {
                url: `https://llvm.org/docs/LangRef.html#resume-instruction`,
                html: `<html><head></head><body><span id="i-resume"></span><h4><a class="toc-backref" href="#id2162" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ Instruction</a></h4>
<section id="id60">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">resume</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id61">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction is a terminator instruction that has no
successors.</p>
</section>
<section id="id62">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction requires one argument, which must have the
same type as the result of any ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction in the same
function.</p>
</section>
<section id="id63">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction resumes propagation of an existing
(in-flight) exception whose unwinding was interrupted with a
<a class="reference internal" href="#i-landingpad"><span class="std std-ref">landingpad</span></a> instruction.</p>
</section>
<section id="id64">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">resume</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="nv">%exn</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘resume’ instruction is a terminator instruction that has nosuccessors.`,
            };
        case 'CATCHSWITCH':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchswitch-instruction`,
                html: `<html><head></head><body><span id="i-catchswitch"></span><h4><a class="toc-backref" href="#id2163" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>’ Instruction</a></h4>
<section id="id65">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchswitch</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span> <span class="n">unwind</span> <span class="n">to</span> <span class="n">caller</span>
<span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchswitch</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">default</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id66">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling system</a> to describe the set of possible catch handlers
that may be executed by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">EH personality routine</span></a>.</p>
</section>
<section id="id67">
<h5>Arguments:</h5>
<p>The <code class="docutils literal notranslate"><span class="pre">parent</span></code> argument is the token of the funclet that contains the
<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> instruction. If the <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> is not inside a funclet,
this operand may be the token <code class="docutils literal notranslate"><span class="pre">none</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">default</span></code> argument is the label of another basic block beginning with
either a <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> or <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> instruction.  This unwind destination
must be a legal target with respect to the <code class="docutils literal notranslate"><span class="pre">parent</span></code> links, as described in
the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">exception handling documentation</a>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">handlers</span></code> are a nonempty list of successor blocks that each begin with a
<a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a> instruction.</p>
</section>
<section id="id68">
<h5>Semantics:</h5>
<p>Executing this instruction transfers control to one of the successors in
<code class="docutils literal notranslate"><span class="pre">handlers</span></code>, if appropriate, or continues to unwind via the unwind label if
present.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> is both a terminator and a “pad” instruction, meaning that
it must be both the first non-phi instruction and last instruction in the basic
block. Therefore, it must be the only non-phi instruction in the block.</p>
</section>
<section id="id69">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>dispatch1:
  %cs1 = catchswitch within none [label %handler0, label %handler1] unwind to caller
dispatch2:
  %cs2 = catchswitch within %parenthandler [label %handler0] unwind label %cleanup
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘catchswitch’ instruction is used by LLVM’s exception handling system to describe the set of possible catch handlersthat may be executed by the EH personality routine.`,
            };
        case 'CATCHRET':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchret-instruction`,
                html: `<html><head></head><body><span id="i-catchret"></span><h4><a class="toc-backref" href="#id2164" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ Instruction</a></h4>
<section id="id70">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">catchret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">token</span><span class="o">&gt;</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">normal</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id71">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ instruction is a terminator instruction that has a
single successor.</p>
</section>
<section id="id72">
<h5>Arguments:</h5>
<p>The first argument to a ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ indicates which <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> it
exits.  It must be a <a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a>.
The second argument to a ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ specifies where control will
transfer to next.</p>
</section>
<section id="id73">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ instruction ends an existing (in-flight) exception whose
unwinding was interrupted with a <a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a> instruction.  The
<a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> gets a chance to execute arbitrary
code to, for example, destroy the active exception.  Control then transfers to
<code class="docutils literal notranslate"><span class="pre">normal</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">token</span></code> argument must be a token produced by a <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> instruction.
If the specified <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> is not the most-recently-entered not-yet-exited
funclet pad (as described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
the <code class="docutils literal notranslate"><span class="pre">catchret</span></code>’s behavior is undefined.</p>
</section>
<section id="id74">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>catchret from %catch to label %continue
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘catchret’ instruction is a terminator instruction that has asingle successor.`,
            };
        case 'CLEANUPRET':
            return {
                url: `https://llvm.org/docs/LangRef.html#cleanupret-instruction`,
                html: `<html><head></head><body><span id="i-cleanupret"></span><h4><a class="toc-backref" href="#id2165" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ Instruction</a></h4>
<section id="id75">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">cleanupret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="k">continue</span><span class="o">&gt;</span>
<span class="n">cleanupret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">to</span> <span class="n">caller</span>
</pre></div>
</div>
</section>
<section id="id76">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction is a terminator instruction that has
an optional successor.</p>
</section>
<section id="id77">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction requires one argument, which indicates
which <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> it exits, and must be a <a class="reference internal" href="#i-cleanuppad"><span class="std std-ref">cleanuppad</span></a>.
If the specified <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> is not the most-recently-entered not-yet-exited
funclet pad (as described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
the <code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’s behavior is undefined.</p>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction also has an optional successor, <code class="docutils literal notranslate"><span class="pre">continue</span></code>,
which must be the label of another basic block beginning with either a
<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> or <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> instruction.  This unwind destination must
be a legal target with respect to the <code class="docutils literal notranslate"><span class="pre">parent</span></code> links, as described in the
<a class="reference external" href="ExceptionHandling.html#wineh-constraints">exception handling documentation</a>.</p>
</section>
<section id="id80">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction indicates to the
<a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> that one
<a class="reference internal" href="#i-cleanuppad"><span class="std std-ref">cleanuppad</span></a> it transferred control to has ended.
It transfers control to <code class="docutils literal notranslate"><span class="pre">continue</span></code> or unwinds out of the function.</p>
</section>
<section id="id81">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>cleanupret from %cleanup unwind to caller
cleanupret from %cleanup unwind label %continue
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘cleanupret’ instruction is a terminator instruction that hasan optional successor.`,
            };
        case 'UNREACHABLE':
            return {
                url: `https://llvm.org/docs/LangRef.html#unreachable-instruction`,
                html: `<html><head></head><body><span id="i-unreachable"></span><h4><a class="toc-backref" href="#id2166" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ Instruction</a></h4>
<section id="id82">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">unreachable</span>
</pre></div>
</div>
</section>
<section id="id83">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ instruction has no defined semantics. This
instruction is used to inform the optimizer that a particular portion of
the code is not reachable. This can be used to indicate that the code
after a no-return function cannot be reached, and other facts.</p>
</section>
<section id="id84">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ instruction has no defined semantics.</p>
</section>
</body></html>`,
                tooltip: `The ‘unreachable’ instruction has no defined semantics. Thisinstruction is used to inform the optimizer that a particular portion ofthe code is not reachable. This can be used to indicate that the codeafter a no-return function cannot be reached, and other facts.`,
            };
        case 'FNEG':
            return {
                url: `https://llvm.org/docs/LangRef.html#fneg-instruction`,
                html: `<html><head></head><body><span id="i-fneg"></span><h4><a class="toc-backref" href="#id2168" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ Instruction</a></h4>
<section id="id85">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fneg</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id86">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ instruction returns the negation of its operand.</p>
</section>
<section id="id87">
<h5>Arguments:</h5>
<p>The argument to the ‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ instruction must be a
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values.</p>
</section>
<section id="id88">
<h5>Semantics:</h5>
<p>The value produced is a copy of the operand with its sign bit flipped.
The value is otherwise completely identical; in particular, if the input is a
NaN, then the quiet/signaling bit and payload are perfectly preserved.</p>
<p>This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id89">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fneg float %val          ; yields float:result = -%var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fneg’ instruction returns the negation of its operand.`,
            };
        case 'ADD':
            return {
                url: `https://llvm.org/docs/LangRef.html#add-instruction`,
                html: `<html><head></head><body><span id="i-add"></span><h4><a class="toc-backref" href="#id2170" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ Instruction</a></h4>
<section id="id90">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id91">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ instruction returns the sum of its two operands.</p>
</section>
<section id="id92">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id93">
<h5>Semantics:</h5>
<p>The value produced is the integer sum of the two operands.</p>
<p>If the sum has unsigned overflow, the result returned is the
mathematical result modulo 2<sup>n</sup>, where n is the bit width of
the result.</p>
<p>Because LLVM integers use a two’s complement representation, this
instruction is appropriate for both signed and unsigned integers.</p>
<p><code class="docutils literal notranslate"><span class="pre">nuw</span></code> and <code class="docutils literal notranslate"><span class="pre">nsw</span></code> stand for “No Unsigned Wrap” and “No Signed Wrap”,
respectively. If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> and/or <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keywords are present, the
result value of the <code class="docutils literal notranslate"><span class="pre">add</span></code> is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if
unsigned and/or signed overflow, respectively, occurs.</p>
</section>
<section id="id94">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = add i32 4, %var          ; yields i32:result = 4 + %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘add’ instruction returns the sum of its two operands.`,
            };
        case 'FADD':
            return {
                url: `https://llvm.org/docs/LangRef.html#fadd-instruction`,
                html: `<html><head></head><body><span id="i-fadd"></span><h4><a class="toc-backref" href="#id2171" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ Instruction</a></h4>
<section id="id95">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fadd</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id96">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ instruction returns the sum of its two operands.</p>
</section>
<section id="id97">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</section>
<section id="id98">
<h5>Semantics:</h5>
<p>The value produced is the floating-point sum of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id99">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fadd float 4.0, %var          ; yields float:result = 4.0 + %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fadd’ instruction returns the sum of its two operands.`,
            };
        case 'SUB':
            return {
                url: `https://llvm.org/docs/LangRef.html#sub-instruction`,
                html: `<html><head></head><body><span id="i-sub"></span><h4><a class="toc-backref" href="#id2172" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ Instruction</a></h4>
<section id="id100">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id101">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction returns the difference of its two operands.</p>
<p>Note that the ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction is used to represent the ‘<code class="docutils literal notranslate"><span class="pre">neg</span></code>’
instruction present in most other intermediate representations.</p>
</section>
<section id="id102">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id103">
<h5>Semantics:</h5>
<p>The value produced is the integer difference of the two operands.</p>
<p>If the difference has unsigned overflow, the result returned is the
mathematical result modulo 2<sup>n</sup>, where n is the bit width of
the result.</p>
<p>Because LLVM integers use a two’s complement representation, this
instruction is appropriate for both signed and unsigned integers.</p>
<p><code class="docutils literal notranslate"><span class="pre">nuw</span></code> and <code class="docutils literal notranslate"><span class="pre">nsw</span></code> stand for “No Unsigned Wrap” and “No Signed Wrap”,
respectively. If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> and/or <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keywords are present, the
result value of the <code class="docutils literal notranslate"><span class="pre">sub</span></code> is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if
unsigned and/or signed overflow, respectively, occurs.</p>
</section>
<section id="id104">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = sub i32 4, %var          ; yields i32:result = 4 - %var
&lt;result&gt; = sub i32 0, %val          ; yields i32:result = -%var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘sub’ instruction returns the difference of its two operands.Note that the ‘sub’ instruction is used to represent the ‘neg’instruction present in most other intermediate representations.`,
            };
        case 'FSUB':
            return {
                url: `https://llvm.org/docs/LangRef.html#fsub-instruction`,
                html: `<html><head></head><body><span id="i-fsub"></span><h4><a class="toc-backref" href="#id2173" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ Instruction</a></h4>
<section id="id105">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fsub</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id106">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ instruction returns the difference of its two operands.</p>
</section>
<section id="id107">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</section>
<section id="id108">
<h5>Semantics:</h5>
<p>The value produced is the floating-point difference of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id109">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fsub float 4.0, %var           ; yields float:result = 4.0 - %var
&lt;result&gt; = fsub float -0.0, %val          ; yields float:result = -%var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fsub’ instruction returns the difference of its two operands.`,
            };
        case 'MUL':
            return {
                url: `https://llvm.org/docs/LangRef.html#mul-instruction`,
                html: `<html><head></head><body><span id="i-mul"></span><h4><a class="toc-backref" href="#id2174" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ Instruction</a></h4>
<section id="id110">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id111">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ instruction returns the product of its two operands.</p>
</section>
<section id="id112">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id113">
<h5>Semantics:</h5>
<p>The value produced is the integer product of the two operands.</p>
<p>If the result of the multiplication has unsigned overflow, the result
returned is the mathematical result modulo 2<sup>n</sup>, where n is the
bit width of the result.</p>
<p>Because LLVM integers use a two’s complement representation, and the
result is the same width as the operands, this instruction returns the
correct result for both signed and unsigned integers. If a full product
(e.g., <code class="docutils literal notranslate"><span class="pre">i32</span></code> * <code class="docutils literal notranslate"><span class="pre">i32</span></code> -&gt; <code class="docutils literal notranslate"><span class="pre">i64</span></code>) is needed, the operands should be
sign-extended or zero-extended as appropriate to the width of the full
product.</p>
<p><code class="docutils literal notranslate"><span class="pre">nuw</span></code> and <code class="docutils literal notranslate"><span class="pre">nsw</span></code> stand for “No Unsigned Wrap” and “No Signed Wrap”,
respectively. If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> and/or <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keywords are present, the
result value of the <code class="docutils literal notranslate"><span class="pre">mul</span></code> is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if
unsigned and/or signed overflow, respectively, occurs.</p>
</section>
<section id="id114">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = mul i32 4, %var          ; yields i32:result = 4 * %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘mul’ instruction returns the product of its two operands.`,
            };
        case 'FMUL':
            return {
                url: `https://llvm.org/docs/LangRef.html#fmul-instruction`,
                html: `<html><head></head><body><span id="i-fmul"></span><h4><a class="toc-backref" href="#id2175" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ Instruction</a></h4>
<section id="id115">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fmul</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id116">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ instruction returns the product of its two operands.</p>
</section>
<section id="id117">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</section>
<section id="id118">
<h5>Semantics:</h5>
<p>The value produced is the floating-point product of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id119">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fmul float 4.0, %var          ; yields float:result = 4.0 * %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fmul’ instruction returns the product of its two operands.`,
            };
        case 'UDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#udiv-instruction`,
                html: `<html><head></head><body><span id="i-udiv"></span><h4><a class="toc-backref" href="#id2176" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ Instruction</a></h4>
<section id="id120">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">udiv</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">udiv</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id121">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ instruction returns the quotient of its two operands.</p>
</section>
<section id="id122">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id123">
<h5>Semantics:</h5>
<p>The value produced is the unsigned integer quotient of the two operands.</p>
<p>Note that unsigned integer division and signed integer division are
distinct operations; for signed integer division, use ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’.</p>
<p>Division by zero is undefined behavior. For vectors, if any element
of the divisor is zero, the operation has undefined behavior.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">udiv</span></code> is
a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if %op1 is not a multiple of %op2 (as
such, “((a udiv exact b) mul b) == a”).</p>
</section>
<section id="id124">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = udiv i32 4, %var          ; yields i32:result = 4 / %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘udiv’ instruction returns the quotient of its two operands.`,
            };
        case 'SDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#sdiv-instruction`,
                html: `<html><head></head><body><span id="i-sdiv"></span><h4><a class="toc-backref" href="#id2177" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ Instruction</a></h4>
<section id="id125">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sdiv</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sdiv</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id126">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ instruction returns the quotient of its two operands.</p>
</section>
<section id="id127">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id128">
<h5>Semantics:</h5>
<p>The value produced is the signed integer quotient of the two operands
rounded towards zero.</p>
<p>Note that signed integer division and unsigned integer division are
distinct operations; for unsigned integer division, use ‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’.</p>
<p>Division by zero is undefined behavior. For vectors, if any element
of the divisor is zero, the operation has undefined behavior.
Overflow also leads to undefined behavior; this is a rare case, but can
occur, for example, by doing a 32-bit division of -2147483648 by -1.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">sdiv</span></code> is
a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if the result would be rounded.</p>
</section>
<section id="id129">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = sdiv i32 4, %var          ; yields i32:result = 4 / %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘sdiv’ instruction returns the quotient of its two operands.`,
            };
        case 'FDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#fdiv-instruction`,
                html: `<html><head></head><body><span id="i-fdiv"></span><h4><a class="toc-backref" href="#id2178" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ Instruction</a></h4>
<section id="id130">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fdiv</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id131">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ instruction returns the quotient of its two operands.</p>
</section>
<section id="id132">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</section>
<section id="id133">
<h5>Semantics:</h5>
<p>The value produced is the floating-point quotient of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id134">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fdiv float 4.0, %var          ; yields float:result = 4.0 / %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fdiv’ instruction returns the quotient of its two operands.`,
            };
        case 'UREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#urem-instruction`,
                html: `<html><head></head><body><span id="i-urem"></span><h4><a class="toc-backref" href="#id2179" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ Instruction</a></h4>
<section id="id135">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">urem</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id136">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ instruction returns the remainder from the unsigned
division of its two arguments.</p>
</section>
<section id="id137">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id138">
<h5>Semantics:</h5>
<p>This instruction returns the unsigned integer <em>remainder</em> of a division.
This instruction always performs an unsigned division to get the
remainder.</p>
<p>Note that unsigned integer remainder and signed integer remainder are
distinct operations; for signed integer remainder, use ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’.</p>
<p>Taking the remainder of a division by zero is undefined behavior.
For vectors, if any element of the divisor is zero, the operation has
undefined behavior.</p>
</section>
<section id="id139">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = urem i32 4, %var          ; yields i32:result = 4 % %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘urem’ instruction returns the remainder from the unsigneddivision of its two arguments.`,
            };
        case 'SREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#srem-instruction`,
                html: `<html><head></head><body><span id="i-srem"></span><h4><a class="toc-backref" href="#id2180" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ Instruction</a></h4>
<section id="id140">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">srem</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id141">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ instruction returns the remainder from the signed
division of its two operands. This instruction can also take
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> versions of the values in which case the elements
must be integers.</p>
</section>
<section id="id142">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id143">
<h5>Semantics:</h5>
<p>This instruction returns the <em>remainder</em> of a division (where the result
is either zero or has the same sign as the dividend, <code class="docutils literal notranslate"><span class="pre">op1</span></code>), not the
<em>modulo</em> operator (where the result is either zero or has the same sign
as the divisor, <code class="docutils literal notranslate"><span class="pre">op2</span></code>) of a value. For more information about the
difference, see <a class="reference external" href="http://mathforum.org/dr.math/problems/anne.4.28.99.html">The Math
Forum</a>. For a
table of how this is implemented in various languages, please see
<a class="reference external" href="http://en.wikipedia.org/wiki/Modulo_operation">Wikipedia: modulo
operation</a>.</p>
<p>Note that signed integer remainder and unsigned integer remainder are
distinct operations; for unsigned integer remainder, use ‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’.</p>
<p>Taking the remainder of a division by zero is undefined behavior.
For vectors, if any element of the divisor is zero, the operation has
undefined behavior.
Overflow also leads to undefined behavior; this is a rare case, but can
occur, for example, by taking the remainder of a 32-bit division of
-2147483648 by -1. (The remainder doesn’t actually overflow, but this
rule lets srem be implemented using instructions that return both the
result of the division and the remainder.)</p>
</section>
<section id="id144">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = srem i32 4, %var          ; yields i32:result = 4 % %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘srem’ instruction returns the remainder from the signeddivision of its two operands. This instruction can also takevector versions of the values in which case the elementsmust be integers.`,
            };
        case 'FREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#frem-instruction`,
                html: `<html><head></head><body><span id="i-frem"></span><h4><a class="toc-backref" href="#id2181" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ Instruction</a></h4>
<section id="id145">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">frem</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id146">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ instruction returns the remainder from the division of
its two operands.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>The instruction is implemented as a call to libm’s ‘<code class="docutils literal notranslate"><span class="pre">fmod</span></code>’
for some targets, and using the instruction may thus require linking libm.</p>
</div>
</section>
<section id="id147">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</section>
<section id="id148">
<h5>Semantics:</h5>
<p>The value produced is the floating-point remainder of the two operands.
This is the same output as a libm ‘<code class="docutils literal notranslate"><span class="pre">fmod</span></code>’ function, but without any
possibility of setting <code class="docutils literal notranslate"><span class="pre">errno</span></code>. The remainder has the same sign as the
dividend.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</section>
<section id="id149">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = frem float 4.0, %var          ; yields float:result = 4.0 % %var
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘frem’ instruction returns the remainder from the division ofits two operands.`,
            };
        case 'SHL':
            return {
                url: `https://llvm.org/docs/LangRef.html#shl-instruction`,
                html: `<html><head></head><body><span id="i-shl"></span><h4><a class="toc-backref" href="#id2183" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ Instruction</a></h4>
<section id="id150">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>           <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id151">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ instruction returns the first operand shifted to the left
a specified number of bits.</p>
</section>
<section id="id152">
<h5>Arguments:</h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</section>
<section id="id153">
<h5>Semantics:</h5>
<p>The value produced is <code class="docutils literal notranslate"><span class="pre">op1</span></code> * 2<sup>op2</sup> mod 2<sup>n</sup>,
where <code class="docutils literal notranslate"><span class="pre">n</span></code> is the width of the result. If <code class="docutils literal notranslate"><span class="pre">op2</span></code> is (statically or
dynamically) equal to or larger than the number of bits in
<code class="docutils literal notranslate"><span class="pre">op1</span></code>, this instruction returns a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.
If the arguments are vectors, each vector element of <code class="docutils literal notranslate"><span class="pre">op1</span></code> is shifted
by the corresponding shift amount in <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> keyword is present, then the shift produces a poison
value if it shifts out any non-zero bits.
If the <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keyword is present, then the shift produces a poison
value if it shifts out any bits that disagree with the resultant sign bit.</p>
</section>
<section id="id154">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = shl i32 4, %var   ; yields i32: 4 &lt;&lt; %var
&lt;result&gt; = shl i32 4, 2      ; yields i32: 16
&lt;result&gt; = shl i32 1, 10     ; yields i32: 1024
&lt;result&gt; = shl i32 1, 32     ; undefined
&lt;result&gt; = shl &lt;2 x i32&gt; &lt; i32 1, i32 1&gt;, &lt; i32 1, i32 2&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 2, i32 4&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘shl’ instruction returns the first operand shifted to the lefta specified number of bits.`,
            };
        case 'LSHR':
            return {
                url: `https://llvm.org/docs/LangRef.html#lshr-instruction`,
                html: `<html><head></head><body><span id="i-lshr"></span><h4><a class="toc-backref" href="#id2184" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ Instruction</a></h4>
<section id="id155">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">lshr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">lshr</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id156">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ instruction (logical shift right) returns the first
operand shifted to the right a specified number of bits with zero fill.</p>
</section>
<section id="id157">
<h5>Arguments:</h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</section>
<section id="id158">
<h5>Semantics:</h5>
<p>This instruction always performs a logical shift right operation. The
most significant bits of the result will be filled with zero bits after
the shift. If <code class="docutils literal notranslate"><span class="pre">op2</span></code> is (statically or dynamically) equal to or larger
than the number of bits in <code class="docutils literal notranslate"><span class="pre">op1</span></code>, this instruction returns a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison
value</span></a>. If the arguments are vectors, each vector element
of <code class="docutils literal notranslate"><span class="pre">op1</span></code> is shifted by the corresponding shift amount in <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">lshr</span></code> is
a poison value if any of the bits shifted out are non-zero.</p>
</section>
<section id="id159">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = lshr i32 4, 1   ; yields i32:result = 2
&lt;result&gt; = lshr i32 4, 2   ; yields i32:result = 1
&lt;result&gt; = lshr i8  4, 3   ; yields i8:result = 0
&lt;result&gt; = lshr i8 -2, 1   ; yields i8:result = 0x7F
&lt;result&gt; = lshr i32 1, 32  ; undefined
&lt;result&gt; = lshr &lt;2 x i32&gt; &lt; i32 -2, i32 4&gt;, &lt; i32 1, i32 2&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 0x7FFFFFFF, i32 1&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘lshr’ instruction (logical shift right) returns the firstoperand shifted to the right a specified number of bits with zero fill.`,
            };
        case 'ASHR':
            return {
                url: `https://llvm.org/docs/LangRef.html#ashr-instruction`,
                html: `<html><head></head><body><span id="i-ashr"></span><h4><a class="toc-backref" href="#id2185" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ Instruction</a></h4>
<section id="id160">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ashr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ashr</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id161">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ instruction (arithmetic shift right) returns the first
operand shifted to the right a specified number of bits with sign
extension.</p>
</section>
<section id="id162">
<h5>Arguments:</h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</section>
<section id="id163">
<h5>Semantics:</h5>
<p>This instruction always performs an arithmetic shift right operation,
The most significant bits of the result will be filled with the sign bit
of <code class="docutils literal notranslate"><span class="pre">op1</span></code>. If <code class="docutils literal notranslate"><span class="pre">op2</span></code> is (statically or dynamically) equal to or larger
than the number of bits in <code class="docutils literal notranslate"><span class="pre">op1</span></code>, this instruction returns a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison
value</span></a>. If the arguments are vectors, each vector element
of <code class="docutils literal notranslate"><span class="pre">op1</span></code> is shifted by the corresponding shift amount in <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">ashr</span></code> is
a poison value if any of the bits shifted out are non-zero.</p>
</section>
<section id="id164">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = ashr i32 4, 1   ; yields i32:result = 2
&lt;result&gt; = ashr i32 4, 2   ; yields i32:result = 1
&lt;result&gt; = ashr i8  4, 3   ; yields i8:result = 0
&lt;result&gt; = ashr i8 -2, 1   ; yields i8:result = -1
&lt;result&gt; = ashr i32 1, 32  ; undefined
&lt;result&gt; = ashr &lt;2 x i32&gt; &lt; i32 -2, i32 4&gt;, &lt; i32 1, i32 3&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 -1, i32 0&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘ashr’ instruction (arithmetic shift right) returns the firstoperand shifted to the right a specified number of bits with signextension.`,
            };
        case 'AND':
            return {
                url: `https://llvm.org/docs/LangRef.html#and-instruction`,
                html: `<html><head></head><body><span id="i-and"></span><h4><a class="toc-backref" href="#id2186" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ Instruction</a></h4>
<section id="id165">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">and</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id166">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction returns the bitwise logical and of its two
operands.</p>
</section>
<section id="id167">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id168">
<h5>Semantics:</h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction is:</p>
<table class="docutils align-default">
<tbody>
<tr class="row-odd"><td><p>In0</p></td>
<td><p>In1</p></td>
<td><p>Out</p></td>
</tr>
<tr class="row-even"><td><p>0</p></td>
<td><p>0</p></td>
<td><p>0</p></td>
</tr>
<tr class="row-odd"><td><p>0</p></td>
<td><p>1</p></td>
<td><p>0</p></td>
</tr>
<tr class="row-even"><td><p>1</p></td>
<td><p>0</p></td>
<td><p>0</p></td>
</tr>
<tr class="row-odd"><td><p>1</p></td>
<td><p>1</p></td>
<td><p>1</p></td>
</tr>
</tbody>
</table>
</section>
<section id="id169">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = and i32 4, %var         ; yields i32:result = 4 &amp; %var
&lt;result&gt; = and i32 15, 40          ; yields i32:result = 8
&lt;result&gt; = and i32 4, 8            ; yields i32:result = 0
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘and’ instruction returns the bitwise logical and of its twooperands.`,
            };
        case 'OR':
            return {
                url: `https://llvm.org/docs/LangRef.html#or-instruction`,
                html: `<html><head></head><body><span id="i-or"></span><h4><a class="toc-backref" href="#id2187" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ Instruction</a></h4>
<section id="id170">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">disjoint</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id171">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction returns the bitwise logical inclusive or of its
two operands.</p>
</section>
<section id="id172">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id173">
<h5>Semantics:</h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction is:</p>
<table class="docutils align-default">
<tbody>
<tr class="row-odd"><td><p>In0</p></td>
<td><p>In1</p></td>
<td><p>Out</p></td>
</tr>
<tr class="row-even"><td><p>0</p></td>
<td><p>0</p></td>
<td><p>0</p></td>
</tr>
<tr class="row-odd"><td><p>0</p></td>
<td><p>1</p></td>
<td><p>1</p></td>
</tr>
<tr class="row-even"><td><p>1</p></td>
<td><p>0</p></td>
<td><p>1</p></td>
</tr>
<tr class="row-odd"><td><p>1</p></td>
<td><p>1</p></td>
<td><p>1</p></td>
</tr>
</tbody>
</table>
<p><code class="docutils literal notranslate"><span class="pre">disjoint</span></code> means that for each bit, that bit is zero in at least one of the
inputs. This allows the Or to be treated as an Add since no carry can occur from
any bit. If the disjoint keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">or</span></code> is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if both inputs have a one in the same bit
position. For vectors, only the element containing the bit is poison.</p>
</section>
<section id="id174">
<h5>Example:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">4</span><span class="p">,</span> <span class="o">%</span><span class="n">var</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">4</span> <span class="o">|</span> <span class="o">%</span><span class="n">var</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">15</span><span class="p">,</span> <span class="mi">40</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">47</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">8</span>            <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">12</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘or’ instruction returns the bitwise logical inclusive or of itstwo operands.`,
            };
        case 'XOR':
            return {
                url: `https://llvm.org/docs/LangRef.html#xor-instruction`,
                html: `<html><head></head><body><span id="i-xor"></span><h4><a class="toc-backref" href="#id2188" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ Instruction</a></h4>
<section id="id175">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">xor</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id176">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction returns the bitwise logical exclusive or of
its two operands. The <code class="docutils literal notranslate"><span class="pre">xor</span></code> is used to implement the “one’s
complement” operation, which is the “~” operator in C.</p>
</section>
<section id="id177">
<h5>Arguments:</h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</section>
<section id="id178">
<h5>Semantics:</h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction is:</p>
<table class="docutils align-default">
<tbody>
<tr class="row-odd"><td><p>In0</p></td>
<td><p>In1</p></td>
<td><p>Out</p></td>
</tr>
<tr class="row-even"><td><p>0</p></td>
<td><p>0</p></td>
<td><p>0</p></td>
</tr>
<tr class="row-odd"><td><p>0</p></td>
<td><p>1</p></td>
<td><p>1</p></td>
</tr>
<tr class="row-even"><td><p>1</p></td>
<td><p>0</p></td>
<td><p>1</p></td>
</tr>
<tr class="row-odd"><td><p>1</p></td>
<td><p>1</p></td>
<td><p>0</p></td>
</tr>
</tbody>
</table>
</section>
<section id="id179">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = xor i32 4, %var         ; yields i32:result = 4 ^ %var
&lt;result&gt; = xor i32 15, 40          ; yields i32:result = 39
&lt;result&gt; = xor i32 4, 8            ; yields i32:result = 12
&lt;result&gt; = xor i32 %V, -1          ; yields i32:result = ~%V
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘xor’ instruction returns the bitwise logical exclusive or ofits two operands. The xor is used to implement the “one’scomplement” operation, which is the “~” operator in C.`,
            };
        case 'EXTRACTELEMENT':
            return {
                url: `https://llvm.org/docs/LangRef.html#extractelement-instruction`,
                html: `<html><head></head><body><span id="i-extractelement"></span><h4><a class="toc-backref" href="#id2190" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ Instruction</a></h4>
<section id="id180">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractelement</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractelement</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id181">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ instruction extracts a single scalar element
from a vector at a specified index.</p>
</section>
<section id="id182">
<h5>Arguments:</h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ instruction is a value of
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> type. The second operand is an index indicating
the position from which to extract the element. The index may be a
variable of any integer type, and will be treated as an unsigned integer.</p>
</section>
<section id="id183">
<h5>Semantics:</h5>
<p>The result is a scalar of the same type as the element type of <code class="docutils literal notranslate"><span class="pre">val</span></code>.
Its value is the value at position <code class="docutils literal notranslate"><span class="pre">idx</span></code> of <code class="docutils literal notranslate"><span class="pre">val</span></code>. If <code class="docutils literal notranslate"><span class="pre">idx</span></code>
exceeds the length of <code class="docutils literal notranslate"><span class="pre">val</span></code> for a fixed-length vector, the result is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. For a scalable vector, if the value
of <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the runtime length of the vector, the result is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id184">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = extractelement &lt;4 x i32&gt; %vec, i32 0    ; yields i32
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘extractelement’ instruction extracts a single scalar elementfrom a vector at a specified index.`,
            };
        case 'INSERTELEMENT':
            return {
                url: `https://llvm.org/docs/LangRef.html#insertelement-instruction`,
                html: `<html><head></head><body><span id="i-insertelement"></span><h4><a class="toc-backref" href="#id2191" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ Instruction</a></h4>
<section id="id185">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertelement</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertelement</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
</pre></div>
</div>
</section>
<section id="id186">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ instruction inserts a scalar element into a
vector at a specified index.</p>
</section>
<section id="id187">
<h5>Arguments:</h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ instruction is a value of
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> type. The second operand is a scalar value whose
type must equal the element type of the first operand. The third operand
is an index indicating the position at which to insert the value. The
index may be a variable of any integer type, and will be treated as an
unsigned integer.</p>
</section>
<section id="id188">
<h5>Semantics:</h5>
<p>The result is a vector of the same type as <code class="docutils literal notranslate"><span class="pre">val</span></code>. Its element values
are those of <code class="docutils literal notranslate"><span class="pre">val</span></code> except at position <code class="docutils literal notranslate"><span class="pre">idx</span></code>, where it gets the value
<code class="docutils literal notranslate"><span class="pre">elt</span></code>. If <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the length of <code class="docutils literal notranslate"><span class="pre">val</span></code> for a fixed-length vector,
the result is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. For a scalable vector,
if the value of <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the runtime length of the vector, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id189">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = insertelement &lt;4 x i32&gt; %vec, i32 1, i32 0    ; yields &lt;4 x i32&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘insertelement’ instruction inserts a scalar element into avector at a specified index.`,
            };
        case 'SHUFFLEVECTOR':
            return {
                url: `https://llvm.org/docs/LangRef.html#shufflevector-instruction`,
                html: `<html><head></head><body><span id="i-shufflevector"></span><h4><a class="toc-backref" href="#id2192" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ Instruction</a></h4>
<section id="id190">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shufflevector</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">m</span> <span class="n">x</span> <span class="n">i32</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">mask</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">m</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shufflevector</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="n">v2</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">m</span> <span class="n">x</span> <span class="n">i32</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">mask</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">m</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
</pre></div>
</div>
</section>
<section id="id191">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ instruction constructs a permutation of elements
from two input vectors, returning a vector with the same element type as
the input and length that is the same as the shuffle mask.</p>
</section>
<section id="id192">
<h5>Arguments:</h5>
<p>The first two operands of a ‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ instruction are vectors
with the same type. The third argument is a shuffle mask vector constant
whose element type is <code class="docutils literal notranslate"><span class="pre">i32</span></code>. The mask vector elements must be constant
integers or <code class="docutils literal notranslate"><span class="pre">poison</span></code> values. The result of the instruction is a vector
whose length is the same as the shuffle mask and whose element type is the
same as the element type of the first two operands.</p>
</section>
<section id="id193">
<h5>Semantics:</h5>
<p>The elements of the two input vectors are numbered from left to right
across both of the vectors. For each element of the result vector, the
shuffle mask selects an element from one of the input vectors to copy
to the result. Non-negative elements in the mask represent an index
into the concatenated pair of input vectors.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">poison</span></code> element in the mask vector specifies that the resulting element
is <code class="docutils literal notranslate"><span class="pre">poison</span></code>.
For backwards-compatibility reasons, LLVM temporarily also accepts <code class="docutils literal notranslate"><span class="pre">undef</span></code>
mask elements, which will be interpreted the same way as <code class="docutils literal notranslate"><span class="pre">poison</span></code> elements.
If the shuffle mask selects an <code class="docutils literal notranslate"><span class="pre">undef</span></code> element from one of the input
vectors, the resulting element is <code class="docutils literal notranslate"><span class="pre">undef</span></code>.</p>
<p>For scalable vectors, the only valid mask values at present are
<code class="docutils literal notranslate"><span class="pre">zeroinitializer</span></code>, <code class="docutils literal notranslate"><span class="pre">undef</span></code> and <code class="docutils literal notranslate"><span class="pre">poison</span></code>, since we cannot write all indices as
literals for a vector with a length unknown at compile time.</p>
</section>
<section id="id194">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; %v2,
                        &lt;4 x i32&gt; &lt;i32 0, i32 4, i32 1, i32 5&gt;  ; yields &lt;4 x i32&gt;
&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; poison,
                        &lt;4 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3&gt;  ; yields &lt;4 x i32&gt; - Identity shuffle.
&lt;result&gt; = shufflevector &lt;8 x i32&gt; %v1, &lt;8 x i32&gt; poison,
                        &lt;4 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3&gt;  ; yields &lt;4 x i32&gt;
&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; %v2,
                        &lt;8 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 &gt;  ; yields &lt;8 x i32&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘shufflevector’ instruction constructs a permutation of elementsfrom two input vectors, returning a vector with the same element type asthe input and length that is the same as the shuffle mask.`,
            };
        case 'EXTRACTVALUE':
            return {
                url: `https://llvm.org/docs/LangRef.html#extractvalue-instruction`,
                html: `<html><head></head><body><span id="i-extractvalue"></span><h4><a class="toc-backref" href="#id2194" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ Instruction</a></h4>
<section id="id195">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractvalue</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
</pre></div>
</div>
</section>
<section id="id196">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction extracts the value of a member field
from an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> value.</p>
</section>
<section id="id197">
<h5>Arguments:</h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction is a value of
<a class="reference internal" href="#t-struct"><span class="std std-ref">struct</span></a> or <a class="reference internal" href="#t-array"><span class="std std-ref">array</span></a> type. The other operands are
constant indices to specify which value to extract in a similar manner
as indices in a ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction.</p>
<p>The major differences to <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> indexing are:</p>
<ul class="simple">
<li><p>Since the value being indexed is not a pointer, the first index is
omitted and assumed to be zero.</p></li>
<li><p>At least one index must be specified.</p></li>
<li><p>Not only struct indices but also array indices must be in bounds.</p></li>
</ul>
</section>
<section id="id198">
<h5>Semantics:</h5>
<p>The result is the value at the position in the aggregate specified by
the index operands.</p>
</section>
<section id="id199">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = extractvalue {i32, float} %agg, 0    ; yields i32
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘extractvalue’ instruction extracts the value of a member fieldfrom an aggregate value.`,
            };
        case 'INSERTVALUE':
            return {
                url: `https://llvm.org/docs/LangRef.html#insertvalue-instruction`,
                html: `<html><head></head><body><span id="i-insertvalue"></span><h4><a class="toc-backref" href="#id2195" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ Instruction</a></h4>
<section id="id200">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertvalue</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id201">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ instruction inserts a value into a member field in
an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> value.</p>
</section>
<section id="id202">
<h5>Arguments:</h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ instruction is a value of
<a class="reference internal" href="#t-struct"><span class="std std-ref">struct</span></a> or <a class="reference internal" href="#t-array"><span class="std std-ref">array</span></a> type. The second operand is
a first-class value to insert. The following operands are constant
indices indicating the position at which to insert the value in a
similar manner as indices in a ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction. The value
to insert must have the same type as the value identified by the
indices.</p>
</section>
<section id="id203">
<h5>Semantics:</h5>
<p>The result is an aggregate of the same type as <code class="docutils literal notranslate"><span class="pre">val</span></code>. Its value is
that of <code class="docutils literal notranslate"><span class="pre">val</span></code> except that the value at the position specified by the
indices is that of <code class="docutils literal notranslate"><span class="pre">elt</span></code>.</p>
</section>
<section id="id204">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%agg1</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">insertvalue</span><span class="w"> </span><span class="p">{</span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="p">}</span><span class="w"> </span><span class="k">poison</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="m">0</span><span class="w">              </span><span class="c">; yields {i32 1, float poison}</span>
<span class="nv">%agg2</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">insertvalue</span><span class="w"> </span><span class="p">{</span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="p">}</span><span class="w"> </span><span class="nv">%agg1</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="nv">%val</span><span class="p">,</span><span class="w"> </span><span class="m">1</span><span class="w">          </span><span class="c">; yields {i32 1, float %val}</span>
<span class="nv">%agg3</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">insertvalue</span><span class="w"> </span><span class="p">{</span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="p">{</span><span class="kt">float</span><span class="p">}}</span><span class="w"> </span><span class="k">poison</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="nv">%val</span><span class="p">,</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="m">0</span><span class="w">    </span><span class="c">; yields {i32 poison, {float %val}}</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘insertvalue’ instruction inserts a value into a member field inan aggregate value.`,
            };
        case 'ALLOCA':
            return {
                url: `https://llvm.org/docs/LangRef.html#alloca-instruction`,
                html: `<html><head></head><body><span id="i-alloca"></span><h4><a class="toc-backref" href="#id2197" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ Instruction</a></h4>
<section id="id205">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">alloca</span> <span class="p">[</span><span class="n">inalloca</span><span class="p">]</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="p">[,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">NumElements</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">[,</span> <span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span>     <span class="p">;</span> <span class="n">yields</span> <span class="nb">type</span> <span class="n">addrspace</span><span class="p">(</span><span class="n">num</span><span class="p">)</span><span class="o">*</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id206">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ instruction allocates memory on the stack frame of the
currently executing function, to be automatically released when this
function returns to its caller. If the address space is not explicitly
specified, the default address space 0 is used.</p>
</section>
<section id="id207">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ instruction allocates <code class="docutils literal notranslate"><span class="pre">sizeof(&lt;type&gt;)*NumElements</span></code>
bytes of memory on the runtime stack, returning a pointer of the
appropriate type to the program. If “NumElements” is specified, it is
the number of elements allocated, otherwise “NumElements” is defaulted
to be one.</p>
<p>If a constant alignment is specified, the value result of the
allocation is guaranteed to be aligned to at least that boundary. The
alignment may not be greater than <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>.</p>
<p>The alignment is only optional when parsing textual IR; for in-memory IR,
it is always present. If not specified, the target can choose to align the
allocation on any convenient boundary compatible with the type.</p>
<p>‘<code class="docutils literal notranslate"><span class="pre">type</span></code>’ may be any sized type.</p>
<p>Structs containing scalable vectors cannot be used in allocas unless all
fields are the same scalable vector type (e.g., <code class="docutils literal notranslate"><span class="pre">{&lt;vscale</span> <span class="pre">x</span> <span class="pre">2</span> <span class="pre">x</span> <span class="pre">i32&gt;,</span>
<span class="pre">&lt;vscale</span> <span class="pre">x</span> <span class="pre">2</span> <span class="pre">x</span> <span class="pre">i32&gt;}</span></code> contains the same type while <code class="docutils literal notranslate"><span class="pre">{&lt;vscale</span> <span class="pre">x</span> <span class="pre">2</span> <span class="pre">x</span> <span class="pre">i32&gt;,</span>
<span class="pre">&lt;vscale</span> <span class="pre">x</span> <span class="pre">2</span> <span class="pre">x</span> <span class="pre">i64&gt;}</span></code> doesn’t).</p>
</section>
<section id="id208">
<h5>Semantics:</h5>
<p>Memory is allocated; a pointer is returned. The allocated memory is
uninitialized, and loading from uninitialized memory produces an undefined
value. The operation itself is undefined if there is insufficient stack
space for the allocation.’<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’d memory is automatically released
when the function returns. The ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ instruction is commonly used
to represent automatic variables that must have an address available. When
the function returns (either with the <code class="docutils literal notranslate"><span class="pre">ret</span></code> or <code class="docutils literal notranslate"><span class="pre">resume</span></code> instructions),
the memory is reclaimed. Allocating zero bytes is legal, but the returned
pointer may not be unique. The order in which memory is allocated (ie.,
which way the stack grows) is not specified.</p>
<p>Note that ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ outside of the alloca address space from the
<a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> is meaningful only if the
target has assigned it a semantics. For targets that specify a non-zero alloca
address space in the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a>, the alloca
address space needs to be explicitly specified in the instruction if it is to be
used.</p>
<p>If the returned pointer is used by <a class="reference internal" href="#int-lifestart"><span class="std std-ref">llvm.lifetime.start</span></a>,
the returned object is initially dead.
See <a class="reference internal" href="#int-lifestart"><span class="std std-ref">llvm.lifetime.start</span></a> and
<a class="reference internal" href="#int-lifeend"><span class="std std-ref">llvm.lifetime.end</span></a> for the precise semantics of
lifetime-manipulating intrinsics.</p>
</section>
<section id="id209">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="w">                             </span><span class="c">; yields ptr</span>
<span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">4</span><span class="w">                      </span><span class="c">; yields ptr</span>
<span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">4</span><span class="p">,</span><span class="w"> </span><span class="k">align</span><span class="w"> </span><span class="m">1024</span><span class="w">          </span><span class="c">; yields ptr</span>
<span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="k">align</span><span class="w"> </span><span class="m">1024</span><span class="w">                 </span><span class="c">; yields ptr</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘alloca’ instruction allocates memory on the stack frame of thecurrently executing function, to be automatically released when thisfunction returns to its caller. If the address space is not explicitlyspecified, the default address space 0 is used.`,
            };
        case 'LOAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#load-instruction`,
                html: `<html><head></head><body><span id="i-load"></span><h4><a class="toc-backref" href="#id2198" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">load</span></code>’ Instruction</a></h4>
<section id="id210">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = load [volatile] &lt;ty&gt;, ptr &lt;pointer&gt;[, align &lt;alignment&gt;][, !nontemporal !&lt;nontemp_node&gt;][, !invariant.load !&lt;empty_node&gt;][, !invariant.group !&lt;empty_node&gt;][, !nonnull !&lt;empty_node&gt;][, !dereferenceable !&lt;deref_bytes_node&gt;][, !dereferenceable_or_null !&lt;deref_bytes_node&gt;][, !align !&lt;align_node&gt;][, !noundef !&lt;empty_node&gt;]
&lt;result&gt; = load atomic [volatile] &lt;ty&gt;, ptr &lt;pointer&gt; [syncscope("&lt;target-scope&gt;")] &lt;ordering&gt;, align &lt;alignment&gt; [, !invariant.group !&lt;empty_node&gt;]
!&lt;nontemp_node&gt; = !{ i32 1 }
!&lt;empty_node&gt; = !{}
!&lt;deref_bytes_node&gt; = !{ i64 &lt;dereferenceable_bytes&gt; }
!&lt;align_node&gt; = !{ i64 &lt;value_alignment&gt; }
</pre></div>
</div>
</section>
<section id="id211">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">load</span></code>’ instruction is used to read from memory.</p>
</section>
<section id="id212">
<h5>Arguments:</h5>
<p>The argument to the <code class="docutils literal notranslate"><span class="pre">load</span></code> instruction specifies the memory address from which
to load. The type specified must be a <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type of
known size (i.e., not containing an <a class="reference internal" href="#t-opaque"><span class="std std-ref">opaque structural type</span></a>). If
the <code class="docutils literal notranslate"><span class="pre">load</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not allowed to
modify the number or order of execution of this <code class="docutils literal notranslate"><span class="pre">load</span></code> with other
<a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">load</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">atomic</span></code>, it takes an extra <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> and optional <code class="docutils literal notranslate"><span class="pre">syncscope("&lt;target-scope&gt;")</span></code> argument. The
<code class="docutils literal notranslate"><span class="pre">release</span></code> and <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code> orderings are not valid on <code class="docutils literal notranslate"><span class="pre">load</span></code> instructions.
Atomic loads produce <a class="reference internal" href="#memmodel"><span class="std std-ref">defined</span></a> results when they may see
multiple atomic stores. The type of the pointee must be an integer, pointer,
floating-point, or vector type whose bit width is a power of two greater than
or equal to eight. <code class="docutils literal notranslate"><span class="pre">align</span></code> must be
explicitly specified on atomic loads. Note: if the alignment is not greater or
equal to the size of the <cite>&lt;value&gt;</cite> type, the atomic operation is likely to
require a lock and have poor performance. <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> does not have any
defined semantics for atomic loads.</p>
<p>The optional constant <code class="docutils literal notranslate"><span class="pre">align</span></code> argument specifies the alignment of the
operation (that is, the alignment of the memory address). It is the
responsibility of the code emitter to ensure that the alignment information is
correct. Overestimating the alignment results in undefined behavior.
Underestimating the alignment may produce less efficient code. An alignment of
1 is always safe. The maximum possible alignment is <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>. An alignment
value higher than the size of the loaded type does <em>not</em> imply (without target
specific knowledge) that memory up to the alignment value bytes can be safely
loaded without trapping.</p>
<p>The alignment is only optional when parsing textual IR; for in-memory IR, it is
always present. An omitted <code class="docutils literal notranslate"><span class="pre">align</span></code> argument means that the operation has the
ABI alignment for the target.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;nontemp_node&gt;</span></code> corresponding to a metadata node with one
<code class="docutils literal notranslate"><span class="pre">i32</span></code> entry of value 1. The existence of the <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code>
metadata on the instruction tells the optimizer and code generator
that this load is not expected to be reused in the cache. The code
generator may select special instructions to save cache bandwidth, such
as the <code class="docutils literal notranslate"><span class="pre">MOVNT</span></code> instruction on x86.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!invariant.load</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no
entries. If a load instruction tagged with the <code class="docutils literal notranslate"><span class="pre">!invariant.load</span></code>
metadata is executed, the memory location referenced by the load has
to contain the same value at all points in the program where the
memory location is dereferenceable; otherwise, the behavior is
undefined.</p>
<dl class="simple">
<dt>The optional <code class="docutils literal notranslate"><span class="pre">!invariant.group</span></code> metadata must reference a single metadata name</dt><dd><p><code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no entries.
See <code class="docutils literal notranslate"><span class="pre">invariant.group</span></code> metadata <a class="reference internal" href="#md-invariant-group"><span class="std std-ref">invariant.group</span></a>.</p>
</dd>
</dl>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nonnull</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no
entries. The existence of the <code class="docutils literal notranslate"><span class="pre">!nonnull</span></code> metadata on the
instruction tells the optimizer that the value loaded is known to
never be null. If the value is null at runtime, a poison value is returned
instead.  This is analogous to the <code class="docutils literal notranslate"><span class="pre">nonnull</span></code> attribute on parameters and
return values. This metadata can only be applied to loads of a pointer type.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!dereferenceable</span></code> metadata must reference a single metadata
name <code class="docutils literal notranslate"><span class="pre">&lt;deref_bytes_node&gt;</span></code> corresponding to a metadata node with one <code class="docutils literal notranslate"><span class="pre">i64</span></code>
entry.
See <code class="docutils literal notranslate"><span class="pre">dereferenceable</span></code> metadata <a class="reference internal" href="#md-dereferenceable"><span class="std std-ref">dereferenceable</span></a>.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!dereferenceable_or_null</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;deref_bytes_node&gt;</span></code> corresponding to a metadata node with one
<code class="docutils literal notranslate"><span class="pre">i64</span></code> entry.
See <code class="docutils literal notranslate"><span class="pre">dereferenceable_or_null</span></code> metadata <a class="reference internal" href="#md-dereferenceable-or-null"><span class="std std-ref">dereferenceable_or_null</span></a>.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!align</span></code> metadata must reference a single metadata name
<code class="docutils literal notranslate"><span class="pre">&lt;align_node&gt;</span></code> corresponding to a metadata node with one <code class="docutils literal notranslate"><span class="pre">i64</span></code> entry.
The existence of the <code class="docutils literal notranslate"><span class="pre">!align</span></code> metadata on the instruction tells the
optimizer that the value loaded is known to be aligned to a boundary specified
by the integer value in the metadata node. The alignment must be a power of 2.
This is analogous to the ‘’align’’ attribute on parameters and return values.
This metadata can only be applied to loads of a pointer type. If the returned
value is not appropriately aligned at runtime, a poison value is returned
instead.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!noundef</span></code> metadata must reference a single metadata name
<code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a node with no entries. The existence of
<code class="docutils literal notranslate"><span class="pre">!noundef</span></code> metadata on the instruction tells the optimizer that the value
loaded is known to be <a class="reference internal" href="#welldefinedvalues"><span class="std std-ref">well defined</span></a>.
If the value isn’t well defined, the behavior is undefined. If the <code class="docutils literal notranslate"><span class="pre">!noundef</span></code>
metadata is combined with poison-generating metadata like <code class="docutils literal notranslate"><span class="pre">!nonnull</span></code>,
violation of that metadata constraint will also result in undefined behavior.</p>
</section>
<section id="id213">
<h5>Semantics:</h5>
<p>The location of memory pointed to is loaded. If the value being loaded
is of scalar type then the number of bytes read does not exceed the
minimum number of bytes needed to hold all bits of the type. For
example, loading an <code class="docutils literal notranslate"><span class="pre">i24</span></code> reads at most three bytes. When loading a
value of a type like <code class="docutils literal notranslate"><span class="pre">i20</span></code> with a size that is not an integral number
of bytes, the result is undefined if the value was not originally
written using a store of the same type.
If the value being loaded is of aggregate type, the bytes that correspond to
padding may be accessed but are ignored, because it is impossible to observe
padding from the loaded aggregate value.
If <code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code> is not a well-defined value, the behavior is undefined.</p>
</section>
<section id="id214">
<h5>Examples:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="w">                               </span><span class="c">; yields ptr</span>
<span class="k">store</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">3</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="w">                           </span><span class="c">; yields void</span>
<span class="nv">%val</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">load</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="w">                       </span><span class="c">; yields i32:val = i32 3</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘load’ instruction is used to read from memory.`,
            };
        case 'STORE':
            return {
                url: `https://llvm.org/docs/LangRef.html#store-instruction`,
                html: `<html><head></head><body><span id="i-store"></span><h4><a class="toc-backref" href="#id2199" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">store</span></code>’ Instruction</a></h4>
<section id="id215">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>store [volatile] &lt;ty&gt; &lt;value&gt;, ptr &lt;pointer&gt;[, align &lt;alignment&gt;][, !nontemporal !&lt;nontemp_node&gt;][, !invariant.group !&lt;empty_node&gt;]        ; yields void
store atomic [volatile] &lt;ty&gt; &lt;value&gt;, ptr &lt;pointer&gt; [syncscope("&lt;target-scope&gt;")] &lt;ordering&gt;, align &lt;alignment&gt; [, !invariant.group !&lt;empty_node&gt;] ; yields void
!&lt;nontemp_node&gt; = !{ i32 1 }
!&lt;empty_node&gt; = !{}
</pre></div>
</div>
</section>
<section id="id216">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">store</span></code>’ instruction is used to write to memory.</p>
</section>
<section id="id217">
<h5>Arguments:</h5>
<p>There are two arguments to the <code class="docutils literal notranslate"><span class="pre">store</span></code> instruction: a value to store and an
address at which to store it. The type of the <code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code> operand must be a
pointer to the <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type of the <code class="docutils literal notranslate"><span class="pre">&lt;value&gt;</span></code>
operand. If the <code class="docutils literal notranslate"><span class="pre">store</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not
allowed to modify the number or order of execution of this <code class="docutils literal notranslate"><span class="pre">store</span></code> with other
<a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.  Only values of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> types of known size (i.e., not containing an <a class="reference internal" href="#t-opaque"><span class="std std-ref">opaque
structural type</span></a>) can be stored.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">store</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">atomic</span></code>, it takes an extra <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> and optional <code class="docutils literal notranslate"><span class="pre">syncscope("&lt;target-scope&gt;")</span></code> argument. The
<code class="docutils literal notranslate"><span class="pre">acquire</span></code> and <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code> orderings aren’t valid on <code class="docutils literal notranslate"><span class="pre">store</span></code> instructions.
Atomic loads produce <a class="reference internal" href="#memmodel"><span class="std std-ref">defined</span></a> results when they may see
multiple atomic stores. The type of the pointee must be an integer, pointer,
floating-point, or vector type whose bit width is a power of two greater than
or equal to eight. <code class="docutils literal notranslate"><span class="pre">align</span></code> must be
explicitly specified on atomic stores. Note: if the alignment is not greater or
equal to the size of the <cite>&lt;value&gt;</cite> type, the atomic operation is likely to
require a lock and have poor performance. <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> does not have any
defined semantics for atomic stores.</p>
<p>The optional constant <code class="docutils literal notranslate"><span class="pre">align</span></code> argument specifies the alignment of the
operation (that is, the alignment of the memory address). It is the
responsibility of the code emitter to ensure that the alignment information is
correct. Overestimating the alignment results in undefined behavior.
Underestimating the alignment may produce less efficient code. An alignment of
1 is always safe. The maximum possible alignment is <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>.  An alignment
value higher than the size of the stored type does <em>not</em> imply (without target
specific knowledge) that memory up to the alignment value bytes can be safely
loaded without trapping.</p>
<p>The alignment is only optional when parsing textual IR; for in-memory IR, it is
always present. An omitted <code class="docutils literal notranslate"><span class="pre">align</span></code> argument means that the operation has the
ABI alignment for the target.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> metadata must reference a single metadata
name <code class="docutils literal notranslate"><span class="pre">&lt;nontemp_node&gt;</span></code> corresponding to a metadata node with one <code class="docutils literal notranslate"><span class="pre">i32</span></code> entry
of value 1. The existence of the <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> metadata on the instruction
tells the optimizer and code generator that this load is not expected to
be reused in the cache. The code generator may select special
instructions to save cache bandwidth, such as the <code class="docutils literal notranslate"><span class="pre">MOVNT</span></code> instruction on
x86.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!invariant.group</span></code> metadata must reference a
single metadata name <code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code>. See <code class="docutils literal notranslate"><span class="pre">invariant.group</span></code> metadata.</p>
</section>
<section id="id218">
<h5>Semantics:</h5>
<p>The contents of memory are updated to contain <code class="docutils literal notranslate"><span class="pre">&lt;value&gt;</span></code> at the
location specified by the <code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code> operand. If <code class="docutils literal notranslate"><span class="pre">&lt;value&gt;</span></code> is
of scalar type then the number of bytes written does not exceed the
minimum number of bytes needed to hold all bits of the type. For
example, storing an <code class="docutils literal notranslate"><span class="pre">i24</span></code> writes at most three bytes. When writing a
value of a type like <code class="docutils literal notranslate"><span class="pre">i20</span></code> with a size that is not an integral number
of bytes, it is unspecified what happens to the extra bits that do not
belong to the type, but they will typically be overwritten.
If <code class="docutils literal notranslate"><span class="pre">&lt;value&gt;</span></code> is of aggregate type, padding is filled with
<a class="reference internal" href="#undefvalues"><span class="std std-ref">undef</span></a>.
If <code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code> is not a well-defined value, the behavior is undefined.</p>
</section>
<section id="id219">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i32</span><span class="w">                               </span><span class="c">; yields ptr</span>
<span class="k">store</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">3</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="w">                           </span><span class="c">; yields void</span>
<span class="nv">%val</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">load</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="w">                       </span><span class="c">; yields i32:val = i32 3</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘store’ instruction is used to write to memory.`,
            };
        case 'FENCE':
            return {
                url: `https://llvm.org/docs/LangRef.html#fence-instruction`,
                html: `<html><head></head><body><span id="i-fence"></span><h4><a class="toc-backref" href="#id2200" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ Instruction</a></h4>
<section id="id220">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">fence</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ordering</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">void</span>
</pre></div>
</div>
</section>
<section id="id221">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ instruction is used to introduce happens-before edges
between operations.</p>
</section>
<section id="id222">
<h5>Arguments:</h5>
<p>‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ instructions take an <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> argument which
defines what <em>synchronizes-with</em> edges they add. They can only be given
<code class="docutils literal notranslate"><span class="pre">acquire</span></code>, <code class="docutils literal notranslate"><span class="pre">release</span></code>, <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code>, and <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code> orderings.</p>
</section>
<section id="id223">
<h5>Semantics:</h5>
<p>A fence A which has (at least) <code class="docutils literal notranslate"><span class="pre">release</span></code> ordering semantics
<em>synchronizes with</em> a fence B with (at least) <code class="docutils literal notranslate"><span class="pre">acquire</span></code> ordering
semantics if and only if there exist atomic operations X and Y, both
operating on some atomic object M, such that A is sequenced before X, X
modifies M (either directly or through some side effect of a sequence
headed by X), Y is sequenced before B, and Y observes M. This provides a
<em>happens-before</em> dependency between A and B. Rather than an explicit
<code class="docutils literal notranslate"><span class="pre">fence</span></code>, one (but not both) of the atomic operations X or Y might
provide a <code class="docutils literal notranslate"><span class="pre">release</span></code> or <code class="docutils literal notranslate"><span class="pre">acquire</span></code> (resp.) ordering constraint and
still <em>synchronize-with</em> the explicit <code class="docutils literal notranslate"><span class="pre">fence</span></code> and establish the
<em>happens-before</em> edge.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">fence</span></code> which has <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code> ordering, in addition to having both
<code class="docutils literal notranslate"><span class="pre">acquire</span></code> and <code class="docutils literal notranslate"><span class="pre">release</span></code> semantics specified above, participates in
the global program order of other <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code> operations and/or
fences. Furthermore, the global ordering created by a <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code>
fence must be compatible with the individual total orders of
<code class="docutils literal notranslate"><span class="pre">monotonic</span></code> (or stronger) memory accesses occurring before and after
such a fence. The exact semantics of this interaction are somewhat
complicated, see the C++ standard’s <a class="reference external" href="https://wg21.link/atomics.order">[atomics.order]</a> section for more details.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">fence</span></code> instruction can also take an optional
“<a class="reference internal" href="#syncscope"><span class="std std-ref">syncscope</span></a>” argument.</p>
</section>
<section id="id225">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>fence acquire                                        ; yields void
fence syncscope("singlethread") seq_cst              ; yields void
fence syncscope("agent") seq_cst                     ; yields void
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fence’ instruction is used to introduce happens-before edgesbetween operations.`,
            };
        case 'CMPXCHG':
            return {
                url: `https://llvm.org/docs/LangRef.html#cmpxchg-instruction`,
                html: `<html><head></head><body><span id="i-cmpxchg"></span><h4><a class="toc-backref" href="#id2201" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ Instruction</a></h4>
<section id="id226">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">cmpxchg</span> <span class="p">[</span><span class="n">weak</span><span class="p">]</span> <span class="p">[</span><span class="n">volatile</span><span class="p">]</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">pointer</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">cmp</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">new</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">success</span> <span class="n">ordering</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">failure</span> <span class="n">ordering</span><span class="o">&gt;</span><span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">;</span> <span class="n">yields</span>  <span class="p">{</span> <span class="n">ty</span><span class="p">,</span> <span class="n">i1</span> <span class="p">}</span>
</pre></div>
</div>
</section>
<section id="id227">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ instruction is used to atomically modify memory. It
loads a value in memory and compares it to a given value. If they are
equal, it tries to store a new value into the memory.</p>
</section>
<section id="id228">
<h5>Arguments:</h5>
<p>There are three arguments to the ‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ instruction: an address
to operate on, a value to compare to the value currently be at that
address, and a new value to place at that address if the compared values
are equal. The type of ‘&lt;cmp&gt;’ must be an integer or pointer type whose
bit width is a power of two greater than or equal to eight.
‘&lt;cmp&gt;’ and ‘&lt;new&gt;’ must
have the same type, and the type of ‘&lt;pointer&gt;’ must be a pointer to
that type. If the <code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the
optimizer is not allowed to modify the number or order of execution of
this <code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> with other <a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.</p>
<p>The success and failure <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> arguments specify how this
<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> synchronizes with other atomic operations. Both ordering parameters
must be at least <code class="docutils literal notranslate"><span class="pre">monotonic</span></code>, the failure ordering cannot be either
<code class="docutils literal notranslate"><span class="pre">release</span></code> or <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code>.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> instruction can also take an optional
“<a class="reference internal" href="#syncscope"><span class="std std-ref">syncscope</span></a>” argument.</p>
<p>Note: if the alignment is not greater or equal to the size of the <cite>&lt;value&gt;</cite>
type, the atomic operation is likely to require a lock and have poor
performance.</p>
<p>The alignment is only optional when parsing textual IR; for in-memory IR, it is
always present. If unspecified, the alignment is assumed to be equal to the
size of the ‘&lt;value&gt;’ type. Note that this default alignment assumption is
different from the alignment used for the load/store instructions when align
isn’t specified.</p>
<p>The pointer passed into cmpxchg must have alignment greater than or
equal to the size in memory of the operand.</p>
</section>
<section id="id229">
<h5>Semantics:</h5>
<p>The contents of memory at the location specified by the ‘<code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code>’ operand
is read and compared to ‘<code class="docutils literal notranslate"><span class="pre">&lt;cmp&gt;</span></code>’; if the values are equal, ‘<code class="docutils literal notranslate"><span class="pre">&lt;new&gt;</span></code>’ is
written to the location. The original value at the location is returned,
together with a flag indicating success (true) or failure (false).</p>
<p>If the cmpxchg operation is marked as <code class="docutils literal notranslate"><span class="pre">weak</span></code> then a spurious failure is
permitted: the operation may not write <code class="docutils literal notranslate"><span class="pre">&lt;new&gt;</span></code> even if the comparison
matched.</p>
<p>If the cmpxchg operation is strong (the default), the i1 value is 1 if and only
if the value loaded equals <code class="docutils literal notranslate"><span class="pre">cmp</span></code>.</p>
<p>A successful <code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> is a read-modify-write instruction for the purpose of
identifying release sequences. A failed <code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code> is equivalent to an atomic
load with an ordering parameter determined the second ordering parameter.</p>
</section>
<section id="id230">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">entry:</span>
<span class="w">  </span><span class="nv">%orig</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">load</span><span class="w"> </span><span class="k">atomic</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="w"> </span><span class="k">unordered</span><span class="p">,</span><span class="w"> </span><span class="k">align</span><span class="w"> </span><span class="m">4</span><span class="w">                      </span><span class="c">; yields i32</span>
<span class="w">  </span><span class="k">br</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%loop</span>

<span class="nl">loop:</span>
<span class="w">  </span><span class="nv">%cmp</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">phi</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="nv">%orig</span><span class="p">,</span><span class="w"> </span><span class="nv">%entry</span><span class="w"> </span><span class="p">],</span><span class="w"> </span><span class="p">[</span><span class="nv">%value_loaded</span><span class="p">,</span><span class="w"> </span><span class="nv">%loop</span><span class="p">]</span>
<span class="w">  </span><span class="nv">%squared</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">mul</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%cmp</span><span class="p">,</span><span class="w"> </span><span class="nv">%cmp</span>
<span class="w">  </span><span class="nv">%val_success</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">cmpxchg</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%cmp</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%squared</span><span class="w"> </span><span class="k">acq_rel</span><span class="w"> </span><span class="k">monotonic</span><span class="w"> </span><span class="c">; yields  { i32, i1 }</span>
<span class="w">  </span><span class="nv">%value_loaded</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">extractvalue</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="nv">%val_success</span><span class="p">,</span><span class="w"> </span><span class="m">0</span>
<span class="w">  </span><span class="nv">%success</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">extractvalue</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="nv">%val_success</span><span class="p">,</span><span class="w"> </span><span class="m">1</span>
<span class="w">  </span><span class="k">br</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="nv">%success</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%done</span><span class="p">,</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%loop</span>

<span class="nl">done:</span>
<span class="w">  </span><span class="p">...</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘cmpxchg’ instruction is used to atomically modify memory. Itloads a value in memory and compares it to a given value. If they areequal, it tries to store a new value into the memory.`,
            };
        case 'ATOMICRMW':
            return {
                url: `https://llvm.org/docs/LangRef.html#atomicrmw-instruction`,
                html: `<html><head></head><body><span id="i-atomicrmw"></span><h4><a class="toc-backref" href="#id2202" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ Instruction</a></h4>
<section id="id231">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">atomicrmw</span> <span class="p">[</span><span class="n">volatile</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">operation</span><span class="o">&gt;</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">pointer</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ordering</span><span class="o">&gt;</span><span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span>
</pre></div>
</div>
</section>
<section id="id232">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ instruction is used to atomically modify memory.</p>
</section>
<section id="id233">
<h5>Arguments:</h5>
<p>There are three arguments to the ‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ instruction: an
operation to apply, an address whose value to modify, an argument to the
operation. The operation must be one of the following keywords:</p>
<ul class="simple">
<li><p>xchg</p></li>
<li><p>add</p></li>
<li><p>sub</p></li>
<li><p>and</p></li>
<li><p>nand</p></li>
<li><p>or</p></li>
<li><p>xor</p></li>
<li><p>max</p></li>
<li><p>min</p></li>
<li><p>umax</p></li>
<li><p>umin</p></li>
<li><p>fadd</p></li>
<li><p>fsub</p></li>
<li><p>fmax</p></li>
<li><p>fmin</p></li>
<li><p>fmaximum</p></li>
<li><p>fminimum</p></li>
<li><p>uinc_wrap</p></li>
<li><p>udec_wrap</p></li>
<li><p>usub_cond</p></li>
<li><p>usub_sat</p></li>
</ul>
<p>For most of these operations, the type of ‘&lt;value&gt;’ must be an integer
type whose bit width is a power of two greater than or equal to eight.
For xchg, this
may also be a floating point or a pointer type with the same size constraints
as integers.  For fadd/fsub/fmax/fmin/fmaximum/fminimum, this must be a floating-point
or fixed vector of floating-point type.  The type of the ‘<code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code>’
operand must be a pointer to that type. If the <code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> is marked
as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not allowed to modify the
number or order of execution of this <code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> with other
<a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.</p>
<p>Note: if the alignment is not greater or equal to the size of the <cite>&lt;value&gt;</cite>
type, the atomic operation is likely to require a lock and have poor
performance.</p>
<p>The alignment is only optional when parsing textual IR; for in-memory IR, it is
always present. If unspecified, the alignment is assumed to be equal to the
size of the ‘&lt;value&gt;’ type. Note that this default alignment assumption is
different from the alignment used for the load/store instructions when align
isn’t specified.</p>
<p>An <code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> instruction can also take an optional
“<a class="reference internal" href="#syncscope"><span class="std std-ref">syncscope</span></a>” argument.</p>
</section>
<section id="id234">
<h5>Semantics:</h5>
<p>The contents of memory at the location specified by the ‘<code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code>’
operand are atomically read, modified, and written back. The original
value at the location is returned. The modification is specified by the
operation argument:</p>
<ul class="simple">
<li><p>xchg: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">val</span></code></p></li>
<li><p>add: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">+</span> <span class="pre">val</span></code></p></li>
<li><p>sub: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span></code></p></li>
<li><p>and: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&amp;</span> <span class="pre">val</span></code></p></li>
<li><p>nand: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">~(*ptr</span> <span class="pre">&amp;</span> <span class="pre">val)</span></code></p></li>
<li><p>or: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">|</span> <span class="pre">val</span></code></p></li>
<li><p>xor: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">^</span> <span class="pre">val</span></code></p></li>
<li><p>max: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&gt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using a signed comparison)</p></li>
<li><p>min: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&lt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using a signed comparison)</p></li>
<li><p>umax: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&gt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using an unsigned comparison)</p></li>
<li><p>umin: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&lt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using an unsigned comparison)</p></li>
<li><p>fadd: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">+</span> <span class="pre">val</span></code> (using floating point arithmetic)</p></li>
<li><p>fsub: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span></code> (using floating point arithmetic)</p></li>
<li><p>fmax: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">maxnum(*ptr,</span> <span class="pre">val)</span></code> (match the <cite>llvm.maxnum.*</cite> intrinsic)</p></li>
<li><p>fmin: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">minnum(*ptr,</span> <span class="pre">val)</span></code> (match the <cite>llvm.minnum.*</cite> intrinsic)</p></li>
<li><p>fmaximum: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">maximum(*ptr,</span> <span class="pre">val)</span></code> (match the <cite>llvm.maximum.*</cite> intrinsic)</p></li>
<li><p>fminimum: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">minimum(*ptr,</span> <span class="pre">val)</span></code> (match the <cite>llvm.minimum.*</cite> intrinsic)</p></li>
<li><p>uinc_wrap: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">(*ptr</span> <span class="pre">u&gt;=</span> <span class="pre">val)</span> <span class="pre">?</span> <span class="pre">0</span> <span class="pre">:</span> <span class="pre">(*ptr</span> <span class="pre">+</span> <span class="pre">1)</span></code> (increment value with wraparound to zero when incremented above input value)</p></li>
<li><p>udec_wrap: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">((*ptr</span> <span class="pre">==</span> <span class="pre">0)</span> <span class="pre">||</span> <span class="pre">(*ptr</span> <span class="pre">u&gt;</span> <span class="pre">val))</span> <span class="pre">?</span> <span class="pre">val</span> <span class="pre">:</span> <span class="pre">(*ptr</span> <span class="pre">-</span> <span class="pre">1)</span></code> (decrement with wraparound to input value when decremented below zero).</p></li>
<li><p>usub_cond: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">(*ptr</span> <span class="pre">u&gt;=</span> <span class="pre">val)</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span> <span class="pre">:</span> <span class="pre">*ptr</span></code> (subtract only if no unsigned overflow).</p></li>
<li><p>usub_sat: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">(*ptr</span> <span class="pre">u&gt;=</span> <span class="pre">val)</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span> <span class="pre">:</span> <span class="pre">0</span></code> (subtract with unsigned clamping to zero).</p></li>
</ul>
</section>
<section id="id235">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%old</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">atomicrmw</span><span class="w"> </span><span class="k">add</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="w"> </span><span class="k">acquire</span><span class="w">                        </span><span class="c">; yields i32</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘atomicrmw’ instruction is used to atomically modify memory.`,
            };
        case 'GETELEMENTPTR':
            return {
                url: `https://llvm.org/docs/LangRef.html#getelementptr-instruction`,
                html: `<html><head></head><body><span id="i-getelementptr"></span><h4><a class="toc-backref" href="#id2203" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ Instruction</a></h4>
<section id="id236">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="n">inbounds</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="n">nusw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="n">inrange</span><span class="p">(</span><span class="n">S</span><span class="p">,</span><span class="n">E</span><span class="p">)</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">ptr</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">ptr</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">vector</span> <span class="n">index</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id237">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction is used to get the address of a
subelement of an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> data structure. It performs
address calculation only and does not access memory. The instruction can also
be used to calculate a vector of such addresses.</p>
</section>
<section id="id238">
<h5>Arguments:</h5>
<p>The first argument is always a type used as the basis for the calculations.
The second argument is always a pointer or a vector of pointers, and is the
base address to start from. The remaining arguments are indices
that indicate which of the elements of the aggregate object are indexed.
The interpretation of each index is dependent on the type being indexed
into. The first index always indexes the pointer value given as the
second argument, the second index indexes a value of the type pointed to
(not necessarily the value directly pointed to, since the first index
can be non-zero), etc. The first type indexed into must be a pointer
value, subsequent types can be arrays, vectors, and structs. Note that
subsequent types being indexed into can never be pointers, since that
would require loading the pointer before continuing calculation.</p>
<p>The type of each index argument depends on the type it is indexing into.
When indexing into a (optionally packed) structure, only <code class="docutils literal notranslate"><span class="pre">i32</span></code> integer
<strong>constants</strong> are allowed (when using a vector of indices they must all
be the <strong>same</strong> <code class="docutils literal notranslate"><span class="pre">i32</span></code> integer constant). When indexing into an array,
pointer or vector, integers of any width are allowed, and they are not
required to be constant. These integers are treated as signed values
where relevant.</p>
<p>For example, let’s consider a C code fragment and how it gets compiled
to LLVM:</p>
<div class="highlight-c notranslate"><div class="highlight"><pre><span></span><span class="k">struct</span><span class="w"> </span><span class="nc">RT</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kt">char</span><span class="w"> </span><span class="n">A</span><span class="p">;</span>
<span class="w">  </span><span class="kt">int</span><span class="w"> </span><span class="n">B</span><span class="p">[</span><span class="mi">10</span><span class="p">][</span><span class="mi">20</span><span class="p">];</span>
<span class="w">  </span><span class="kt">char</span><span class="w"> </span><span class="n">C</span><span class="p">;</span>
<span class="p">};</span>
<span class="k">struct</span><span class="w"> </span><span class="nc">ST</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kt">int</span><span class="w"> </span><span class="n">X</span><span class="p">;</span>
<span class="w">  </span><span class="kt">double</span><span class="w"> </span><span class="n">Y</span><span class="p">;</span>
<span class="w">  </span><span class="k">struct</span><span class="w"> </span><span class="nc">RT</span><span class="w"> </span><span class="n">Z</span><span class="p">;</span>
<span class="p">};</span>

<span class="kt">int</span><span class="w"> </span><span class="o">*</span><span class="nf">foo</span><span class="p">(</span><span class="k">struct</span><span class="w"> </span><span class="nc">ST</span><span class="w"> </span><span class="o">*</span><span class="n">s</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="o">&amp;</span><span class="n">s</span><span class="p">[</span><span class="mi">1</span><span class="p">].</span><span class="n">Z</span><span class="p">.</span><span class="n">B</span><span class="p">[</span><span class="mi">5</span><span class="p">][</span><span class="mi">13</span><span class="p">];</span>
<span class="p">}</span>
</pre></div>
</div>
<p>The LLVM code generated by Clang is approximately:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%struct.RT</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">type</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i8</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="m">10</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="p">[</span><span class="m">20</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">]],</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="p">}</span>
<span class="nv">%struct.ST</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">type</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">double</span><span class="p">,</span><span class="w"> </span><span class="nv">%struct.RT</span><span class="w"> </span><span class="p">}</span>

<span class="k">define</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@foo</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%s</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="nv">%arrayidx</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="k">inbounds</span><span class="w"> </span><span class="nv">%struct.ST</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%s</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">5</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%arrayidx</span>
<span class="p">}</span>
</pre></div>
</div>
</section>
<section id="id239">
<h5>Semantics:</h5>
<p>In the example above, the first index is indexing into the
‘<code class="docutils literal notranslate"><span class="pre">%struct.ST*</span></code>’ type, which is a pointer, yielding a ‘<code class="docutils literal notranslate"><span class="pre">%struct.ST</span></code>’
= ‘<code class="docutils literal notranslate"><span class="pre">{</span> <span class="pre">i32,</span> <span class="pre">double,</span> <span class="pre">%struct.RT</span> <span class="pre">}</span></code>’ type, a structure. The second index
indexes into the third element of the structure, yielding a
‘<code class="docutils literal notranslate"><span class="pre">%struct.RT</span></code>’ = ‘<code class="docutils literal notranslate"><span class="pre">{</span> <span class="pre">i8</span> <span class="pre">,</span> <span class="pre">[10</span> <span class="pre">x</span> <span class="pre">[20</span> <span class="pre">x</span> <span class="pre">i32]],</span> <span class="pre">i8</span> <span class="pre">}</span></code>’ type, another
structure. The third index indexes into the second element of the
structure, yielding a ‘<code class="docutils literal notranslate"><span class="pre">[10</span> <span class="pre">x</span> <span class="pre">[20</span> <span class="pre">x</span> <span class="pre">i32]]</span></code>’ type, an array. The two
dimensions of the array are subscripted into, yielding an ‘<code class="docutils literal notranslate"><span class="pre">i32</span></code>’
type. The ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction returns a pointer to this
element.</p>
<p>Note that it is perfectly legal to index partially through a structure,
returning a pointer to an inner element. Because of this, the LLVM code
for the given testcase is equivalent to:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">define</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@foo</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%s</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="nv">%t1</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="nv">%struct.ST</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%s</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="w">  </span><span class="nv">%t2</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="nv">%struct.ST</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%t1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span>
<span class="w">  </span><span class="nv">%t3</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="nv">%struct.RT</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%t2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="w">  </span><span class="nv">%t4</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">[</span><span class="m">10</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="p">[</span><span class="m">20</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">]],</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%t3</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">5</span>
<span class="w">  </span><span class="nv">%t5</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">[</span><span class="m">20</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">],</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%t4</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">13</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%t5</span>
<span class="p">}</span>
</pre></div>
</div>
<p>The indices are first converted to offsets in the pointer’s index type. If the
currently indexed type is a struct type, the struct offset corresponding to the
index is sign-extended or truncated to the pointer index type. Otherwise, the
index itself is sign-extended or truncated, and then multiplied by the type
allocation size (that is, the size rounded up to the ABI alignment) of the
currently indexed type.</p>
<p>The offsets are then added to the low bits of the base address up to the index
type width, with silently-wrapping two’s complement arithmetic. If the pointer
size is larger than the index size, this means that the bits outside the index
type width will not be affected.</p>
<p>The result value of the <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> may be outside the object pointed
to by the base pointer. The result value may not necessarily be used to access
memory though, even if it happens to point into allocated storage. See the
<a class="reference internal" href="#pointeraliasing"><span class="std std-ref">Pointer Aliasing Rules</span></a> section for more
information.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> instruction may have a number of attributes that impose
additional rules. If any of the rules are violated, the result value is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. In cases where the base is a vector of
pointers, the attributes apply to each computation element-wise.</p>
<p>For <code class="docutils literal notranslate"><span class="pre">nusw</span></code> (no unsigned signed wrap):</p>
<blockquote>
<div><ul class="simple">
<li><p>If the type of an index is larger than the pointer index type, the
truncation to the pointer index type preserves the signed value
(<code class="docutils literal notranslate"><span class="pre">trunc</span> <span class="pre">nsw</span></code>).</p></li>
<li><p>The multiplication of an index by the type size does not wrap the pointer
index type in a signed sense (<code class="docutils literal notranslate"><span class="pre">mul</span> <span class="pre">nsw</span></code>).</p></li>
<li><p>The successive addition of each offset (without adding the base address)
does not wrap the pointer index type in a signed sense (<code class="docutils literal notranslate"><span class="pre">add</span> <span class="pre">nsw</span></code>).</p></li>
<li><p>The successive addition of the current address, truncated to the pointer
index type and interpreted as an unsigned number, and each offset,
interpreted as a signed number, does not wrap the pointer index type.</p></li>
</ul>
</div></blockquote>
<p>For <code class="docutils literal notranslate"><span class="pre">nuw</span></code> (no unsigned wrap):</p>
<blockquote>
<div><ul class="simple">
<li><p>If the type of an index is larger than the pointer index type, the
truncation to the pointer index type preserves the unsigned value
(<code class="docutils literal notranslate"><span class="pre">trunc</span> <span class="pre">nuw</span></code>).</p></li>
<li><p>The multiplication of an index by the type size does not wrap the pointer
index type in an unsigned sense (<code class="docutils literal notranslate"><span class="pre">mul</span> <span class="pre">nuw</span></code>).</p></li>
<li><p>The successive addition of each offset (without adding the base address)
does not wrap the pointer index type in an unsigned sense (<code class="docutils literal notranslate"><span class="pre">add</span> <span class="pre">nuw</span></code>).</p></li>
<li><p>The successive addition of the current address, truncated to the pointer
index type and interpreted as an unsigned number, and each offset, also
interpreted as an unsigned number, does not wrap the pointer index type
(<code class="docutils literal notranslate"><span class="pre">add</span> <span class="pre">nuw</span></code>).</p></li>
</ul>
</div></blockquote>
<p>For <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> all rules of the <code class="docutils literal notranslate"><span class="pre">nusw</span></code> attribute apply. Additionally,
if the <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> has any non-zero indices, the following rules apply:</p>
<blockquote>
<div><ul class="simple">
<li><p>The base pointer has an <em>in bounds</em> address of the
<a class="reference internal" href="#allocatedobjects"><span class="std std-ref">allocated object</span></a> that it is
<a class="reference internal" href="#pointeraliasing"><span class="std std-ref">based</span></a> on. This means that it points into that
allocated object, or to its end. Note that the object does not have to be
live anymore; being in-bounds of a deallocated object is sufficient.
If the allocated object can grow, then the relevant size for being <em>in
bounds</em> is the maximal size the object could have while satisfying the
allocated object rules, not its current size.</p></li>
<li><p>During the successive addition of offsets to the address, the resulting
pointer must remain <em>in bounds</em> of the allocated object at each step.</p></li>
</ul>
</div></blockquote>
<p>Note that <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> with all-zero indices is always considered to be
<code class="docutils literal notranslate"><span class="pre">inbounds</span></code>, even if the base pointer does not point to an allocated object.
As a corollary, the only pointer in bounds of the null pointer in the default
address space is the null pointer itself.</p>
<p>If <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> is present on a <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> instruction, the <code class="docutils literal notranslate"><span class="pre">nusw</span></code>
attribute will be automatically set as well. For this reason, the <code class="docutils literal notranslate"><span class="pre">nusw</span></code>
will also not be printed in textual IR if <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> is already present.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">inrange(Start,</span> <span class="pre">End)</span></code> attribute is present, loading from or
storing to any pointer derived from the <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> has undefined
behavior if the load or store would access memory outside the half-open range
<code class="docutils literal notranslate"><span class="pre">[Start,</span> <span class="pre">End)</span></code> from the <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> expression result. The result of
a pointer comparison or <code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code> (including <code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>-like operations
involving memory) involving a pointer derived from a <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> with
the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword is undefined, with the exception of comparisons
in the case where both operands are in the closed range <code class="docutils literal notranslate"><span class="pre">[Start,</span> <span class="pre">End]</span></code>.
Note that the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword is currently only allowed
in constant <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> expressions.</p>
<p>The getelementptr instruction is often confusing. For some more insight
into how it works, see <a class="reference internal" href="GetElementPtr.html"><span class="doc">the getelementptr FAQ</span></a>.</p>
</section>
<section id="id240">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%aptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">{</span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="p">[</span><span class="m">12</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i8</span><span class="p">]},</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%saptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="nv">%vptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">{</span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i8</span><span class="p">&gt;},</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%svptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="nv">%eptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">[</span><span class="m">12</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i8</span><span class="p">],</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%aptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span>
<span class="nv">%iptr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="p">[</span><span class="m">10</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">],</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@arr</span><span class="p">,</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="m">0</span>
</pre></div>
</div>
</section>
<section id="vector-of-pointers">
<h5>Vector of pointers:</h5>
<p>The <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> returns a vector of pointers, instead of a single address,
when one or more of its arguments is a vector. In such cases, all vector
arguments should have the same number of elements, and every scalar argument
will be effectively broadcast into a vector during address calculation.</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; All arguments are vectors:</span>
<span class="c">;   A[i] = ptrs[i] + offsets[i]*sizeof(i8)</span>
<span class="nv">%A</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="kt">i8</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i8</span><span class="p">*&gt;</span><span class="w"> </span><span class="nv">%ptrs</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%offsets</span>

<span class="c">; Add the same scalar offset to each pointer of a vector:</span>
<span class="c">;   A[i] = ptrs[i] + offset*sizeof(i8)</span>
<span class="nv">%A</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="kt">i8</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%ptrs</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="nv">%offset</span>

<span class="c">; Add distinct offsets to the same pointer:</span>
<span class="c">;   A[i] = ptr + offsets[i]*sizeof(i8)</span>
<span class="nv">%A</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="kt">i8</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%ptr</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%offsets</span>

<span class="c">; In all cases described above the type of the result is &lt;4 x ptr&gt;</span>
</pre></div>
</div>
<p>The two following instructions are equivalent:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">getelementptr</span><span class="w">  </span><span class="nv">%struct.ST</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%s</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%ind1</span><span class="p">,</span>
<span class="w">  </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">&gt;,</span>
<span class="w">  </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">&gt;,</span>
<span class="w">  </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%ind4</span><span class="p">,</span>
<span class="w">  </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span><span class="p">&gt;</span>

<span class="k">getelementptr</span><span class="w">  </span><span class="nv">%struct.ST</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%s</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%ind1</span><span class="p">,</span>
<span class="w">  </span><span class="kt">i32</span><span class="w"> </span><span class="m">2</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">1</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%ind4</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">13</span>
</pre></div>
</div>
<p>Let’s look at the C code, where the vector version of <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>
makes sense:</p>
<div class="highlight-c notranslate"><div class="highlight"><pre><span></span><span class="c1">// Let's assume that we vectorize the following loop:</span>
<span class="kt">double</span><span class="w"> </span><span class="o">*</span><span class="n">A</span><span class="p">,</span><span class="w"> </span><span class="o">*</span><span class="n">B</span><span class="p">;</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="o">*</span><span class="n">C</span><span class="p">;</span>
<span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">size</span><span class="p">;</span><span class="w"> </span><span class="o">++</span><span class="n">i</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">A</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">B</span><span class="p">[</span><span class="n">C</span><span class="p">[</span><span class="n">i</span><span class="p">]];</span>
<span class="p">}</span>
</pre></div>
</div>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; get pointers for 8 elements from array B</span>
<span class="nv">%ptrs</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">getelementptr</span><span class="w"> </span><span class="kt">double</span><span class="p">,</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%B</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">8</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%C</span>
<span class="c">; load 8 elements from array B into A</span>
<span class="nv">%A</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="p">&lt;</span><span class="m">8</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">double</span><span class="p">&gt;</span><span class="w"> </span><span class="vg">@llvm.masked.gather.v8f64.v8p0f64</span><span class="p">(</span>
<span class="w">     </span><span class="p">&lt;</span><span class="m">8</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="k">align</span><span class="w"> </span><span class="m">8</span><span class="w"> </span><span class="nv">%ptrs</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">8</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i1</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%mask</span><span class="p">,</span><span class="w"> </span><span class="p">&lt;</span><span class="m">8</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">double</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%passthru</span><span class="p">)</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘getelementptr’ instruction is used to get the address of asubelement of an aggregate data structure. It performsaddress calculation only and does not access memory. The instruction can alsobe used to calculate a vector of such addresses.`,
            };
        case 'TRUNC':
            return {
                url: `https://llvm.org/docs/LangRef.html#trunc-to-instruction`,
                html: `<html><head></head><body><span id="i-trunc"></span><h4><a class="toc-backref" href="#id2205" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">trunc</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id241">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">trunc</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">trunc</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">trunc</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">trunc</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>     <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id242">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction truncates its operand to the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id243">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction takes a value to trunc, and a type to trunc
it to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors
of the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
larger than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Equal sized
types are not allowed.</p>
</section>
<section id="id244">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction truncates the high order bits in <code class="docutils literal notranslate"><span class="pre">value</span></code>
and converts the remaining bits to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Since the source size must
be larger than the destination size, <code class="docutils literal notranslate"><span class="pre">trunc</span></code> cannot be a <em>no-op cast</em>.
It will always truncate bits.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> keyword is present, and any of the truncated bits are non-zero,
the result is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. If the <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keyword
is present, and any of the truncated bits are not the same as the top bit
of the truncation result, the result is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id245">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">trunc</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">257</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i8</span><span class="w">                        </span><span class="c">; yields i8:1</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">trunc</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">123</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i1</span><span class="w">                        </span><span class="c">; yields i1:true</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">trunc</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">122</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i1</span><span class="w">                        </span><span class="c">; yields i1:false</span>
<span class="nv">%W</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">trunc</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i16</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i16</span><span class="w"> </span><span class="m">8</span><span class="p">,</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="m">7</span><span class="p">&gt;</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i8</span><span class="p">&gt;</span><span class="w"> </span><span class="c">; yields &lt;i8 8, i8 7&gt;</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘trunc’ instruction truncates its operand to the type ty2.`,
            };
        case 'ZEXT':
            return {
                url: `https://llvm.org/docs/LangRef.html#zext-to-instruction`,
                html: `<html><head></head><body><span id="i-zext"></span><h4><a class="toc-backref" href="#id2206" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">zext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id246">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">zext</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id247">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">zext</span></code>’ instruction zero extends its operand to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">nneg</span></code> (non-negative) flag, if present, specifies that the operand is
non-negative. This property may be used by optimization passes to later
convert the <code class="docutils literal notranslate"><span class="pre">zext</span></code> into a <code class="docutils literal notranslate"><span class="pre">sext</span></code>.</p>
</section>
<section id="id248">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">zext</span></code>’ instruction takes a value to cast, and a type to cast it
to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors of
the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
smaller than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id249">
<h5>Semantics:</h5>
<p>The <code class="docutils literal notranslate"><span class="pre">zext</span></code> fills the high order bits of the <code class="docutils literal notranslate"><span class="pre">value</span></code> with zero bits
until it reaches the size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
<p>When zero extending from i1, the result will always be either 0 or 1.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">nneg</span></code> flag is set, and the <code class="docutils literal notranslate"><span class="pre">zext</span></code> argument is negative, the result
is a poison value.</p>
</section>
<section id="id250">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">257</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i64</span><span class="w">              </span><span class="c">; yields i64:257</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="k">true</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">              </span><span class="c">; yields i32:1</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i16</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i16</span><span class="w"> </span><span class="m">8</span><span class="p">,</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="m">7</span><span class="p">&gt;</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="c">; yields &lt;i32 8, i32 7&gt;</span>

<span class="nv">%a</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="err">nneg</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">127</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="c">; yields i16 127</span>
<span class="nv">%b</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">zext</span><span class="w"> </span><span class="err">nneg</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">-1</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i16</span><span class="w">  </span><span class="c">; yields i16 poison</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘zext’ instruction zero extends its operand to type ty2.The nneg (non-negative) flag, if present, specifies that the operand isnon-negative. This property may be used by optimization passes to laterconvert the zext into a sext.`,
            };
        case 'SEXT':
            return {
                url: `https://llvm.org/docs/LangRef.html#sext-to-instruction`,
                html: `<html><head></head><body><span id="i-sext"></span><h4><a class="toc-backref" href="#id2207" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">sext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id251">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sext</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id252">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ sign extends <code class="docutils literal notranslate"><span class="pre">value</span></code> to the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id253">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ instruction takes a value to cast, and a type to cast it
to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors of
the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
smaller than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id254">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ instruction performs a sign extension by copying the sign
bit (highest order bit) of the <code class="docutils literal notranslate"><span class="pre">value</span></code> until it reaches the bit size
of the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
<p>When sign extending from i1, the extension always results in -1 or 0.</p>
</section>
<section id="id255">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">sext</span><span class="w"> </span><span class="kt">i8</span><span class="w">  </span><span class="m">-1</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i16</span><span class="w">              </span><span class="c">; yields i16   :65535</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">sext</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="k">true</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">             </span><span class="c">; yields i32:-1</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">sext</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i16</span><span class="p">&gt;</span><span class="w"> </span><span class="p">&lt;</span><span class="kt">i16</span><span class="w"> </span><span class="m">8</span><span class="p">,</span><span class="w"> </span><span class="kt">i16</span><span class="w"> </span><span class="m">7</span><span class="p">&gt;</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">2</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="c">; yields &lt;i32 8, i32 7&gt;</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘sext’ sign extends value to the type ty2.`,
            };
        case 'FPTRUNC':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptrunc-to-instruction`,
                html: `<html><head></head><body><span id="i-fptrunc"></span><h4><a class="toc-backref" href="#id2208" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id256">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptrunc</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id257">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction truncates <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id258">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction takes a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
value to cast and a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to cast it to.
The size of <code class="docutils literal notranslate"><span class="pre">value</span></code> must be larger than the size of <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. This
implies that <code class="docutils literal notranslate"><span class="pre">fptrunc</span></code> cannot be used to make a <em>no-op cast</em>.</p>
</section>
<section id="id259">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction casts a <code class="docutils literal notranslate"><span class="pre">value</span></code> from a larger
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to a smaller <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.</p>
<p>NaN values follow the usual <a class="reference internal" href="#floatnan"><span class="std std-ref">NaN behaviors</span></a>, except that _if_ a
NaN payload is propagated from the input (“Quieting NaN propagation” or
“Unchanged NaN propagation” cases), then the low order bits of the NaN payload
which cannot fit in the resulting type are discarded. Note that if discarding
the low order bits leads to an all-0 payload, this cannot be represented as a
signaling NaN (it would represent an infinity instead), so in that case
“Unchanged NaN propagation” is not possible.</p>
<p>This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations.</p>
</section>
<section id="id260">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptrunc</span><span class="w"> </span><span class="kt">double</span><span class="w"> </span><span class="m">16777217.0</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">float</span><span class="w">    </span><span class="c">; yields float:16777216.0</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptrunc</span><span class="w"> </span><span class="kt">double</span><span class="w"> </span><span class="m">1.0E+300</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">half</span><span class="w">       </span><span class="c">; yields half:+infinity</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fptrunc’ instruction truncates value to type ty2.`,
            };
        case 'FPEXT':
            return {
                url: `https://llvm.org/docs/LangRef.html#fpext-to-instruction`,
                html: `<html><head></head><body><span id="i-fpext"></span><h4><a class="toc-backref" href="#id2209" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fpext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id261">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fpext</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id262">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ extends a floating-point <code class="docutils literal notranslate"><span class="pre">value</span></code> to a larger floating-point
value.</p>
</section>
<section id="id263">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ instruction takes a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
<code class="docutils literal notranslate"><span class="pre">value</span></code> to cast, and a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to cast it
to. The source type must be smaller than the destination type.</p>
</section>
<section id="id264">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ instruction extends the <code class="docutils literal notranslate"><span class="pre">value</span></code> from a smaller
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to a larger <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. The <code class="docutils literal notranslate"><span class="pre">fpext</span></code> cannot be used to make a
<em>no-op cast</em> because it always changes bits. Use <code class="docutils literal notranslate"><span class="pre">bitcast</span></code> to make a
<em>no-op cast</em> for a floating-point cast.</p>
<p>NaN values follow the usual <a class="reference internal" href="#floatnan"><span class="std std-ref">NaN behaviors</span></a>, except that _if_ a
NaN payload is propagated from the input (“Quieting NaN propagation” or
“Unchanged NaN propagation” cases), then it is copied to the high order bits of
the resulting payload, and the remaining low order bits are zero.</p>
<p>This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations.</p>
</section>
<section id="id265">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fpext</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">3.125</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">double</span><span class="w">         </span><span class="c">; yields double:3.125000e+00</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fpext</span><span class="w"> </span><span class="kt">double</span><span class="w"> </span><span class="nv">%X</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">fp128</span><span class="w">            </span><span class="c">; yields fp128:0xL00000000000000004000900000000000</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fpext’ extends a floating-point value to a larger floating-pointvalue.`,
            };
        case 'FPTOUI':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptoui-to-instruction`,
                html: `<html><head></head><body><h4><a class="toc-backref" href="#id2210" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fptoui</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id266">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptoui</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id267">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ converts a floating-point <code class="docutils literal notranslate"><span class="pre">value</span></code> to its unsigned
integer equivalent of type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id268">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> value, and a type to
cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector floating-point type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector integer
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</section>
<section id="id269">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ instruction converts its <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> operand into the nearest (rounding towards zero)
unsigned integer value. If the value cannot fit in <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id270">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptoui</span><span class="w"> </span><span class="kt">double</span><span class="w"> </span><span class="m">123.0</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">      </span><span class="c">; yields i32:123</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptoui</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">1.0E+300</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i1</span><span class="w">     </span><span class="c">; yields undefined:1</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptoui</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">1.04E+17</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i8</span><span class="w">     </span><span class="c">; yields undefined:1</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fptoui’ converts a floating-point value to its unsignedinteger equivalent of type ty2.`,
            };
        case 'FPTOSI':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptosi-to-instruction`,
                html: `<html><head></head><body><h4><a class="toc-backref" href="#id2211" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fptosi</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id271">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptosi</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id272">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction converts <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
<code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id273">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> value, and a type to
cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector floating-point type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector integer
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</section>
<section id="id274">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction converts its <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> operand into the nearest (rounding towards zero)
signed integer value. If the value cannot fit in <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id275">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptosi</span><span class="w"> </span><span class="kt">double</span><span class="w"> </span><span class="m">-123.0</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">      </span><span class="c">; yields i32:-123</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptosi</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">1.0E-247</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i1</span><span class="w">      </span><span class="c">; yields undefined:1</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">fptosi</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">1.04E+17</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i8</span><span class="w">      </span><span class="c">; yields undefined:1</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fptosi’ instruction converts floating-pointvalue to type ty2.`,
            };
        case 'UITOFP':
            return {
                url: `https://llvm.org/docs/LangRef.html#uitofp-to-instruction`,
                html: `<html><head></head><body><h4><a class="toc-backref" href="#id2212" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">uitofp</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id276">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">uitofp</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id277">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction regards <code class="docutils literal notranslate"><span class="pre">value</span></code> as an unsigned integer
and converts that value to the <code class="docutils literal notranslate"><span class="pre">ty2</span></code> type.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">nneg</span></code> (non-negative) flag, if present, specifies that the
operand is non-negative. This property may be used by optimization
passes to later convert the <code class="docutils literal notranslate"><span class="pre">uitofp</span></code> into a <code class="docutils literal notranslate"><span class="pre">sitofp</span></code>.</p>
</section>
<section id="id278">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> value, and a type to cast it to
<code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector integer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector floating-point
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</section>
<section id="id279">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction interprets its operand as an unsigned
integer quantity and converts it to the corresponding floating-point
value. If the value cannot be exactly represented, it is rounded using
the default rounding mode.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">nneg</span></code> flag is set, and the <code class="docutils literal notranslate"><span class="pre">uitofp</span></code> argument is negative,
the result is a poison value.</p>
</section>
<section id="id280">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">uitofp</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">257</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">float</span><span class="w">         </span><span class="c">; yields float:257.0</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">uitofp</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">-1</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">double</span><span class="w">          </span><span class="c">; yields double:255.0</span>

<span class="nv">%a</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">uitofp</span><span class="w"> </span><span class="err">nneg</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">256</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">      </span><span class="c">; yields float:256.0</span>
<span class="nv">%b</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">uitofp</span><span class="w"> </span><span class="err">nneg</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">-256</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">     </span><span class="c">; yields i32 poison</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘uitofp’ instruction regards value as an unsigned integerand converts that value to the ty2 type.The nneg (non-negative) flag, if present, specifies that theoperand is non-negative. This property may be used by optimizationpasses to later convert the uitofp into a sitofp.`,
            };
        case 'SITOFP':
            return {
                url: `https://llvm.org/docs/LangRef.html#sitofp-to-instruction`,
                html: `<html><head></head><body><h4><a class="toc-backref" href="#id2213" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">sitofp</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id281">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sitofp</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id282">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction regards <code class="docutils literal notranslate"><span class="pre">value</span></code> as a signed integer and
converts that value to the <code class="docutils literal notranslate"><span class="pre">ty2</span></code> type.</p>
</section>
<section id="id283">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> value, and a type to cast it to
<code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector integer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector floating-point
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</section>
<section id="id284">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction interprets its operand as a signed integer
quantity and converts it to the corresponding floating-point value. If the
value cannot be exactly represented, it is rounded using the default rounding
mode.</p>
</section>
<section id="id285">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">sitofp</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">257</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">float</span><span class="w">         </span><span class="c">; yields float:257.0</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">sitofp</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">-1</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">double</span><span class="w">          </span><span class="c">; yields double:-1.0</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘sitofp’ instruction regards value as a signed integer andconverts that value to the ty2 type.`,
            };
        case 'PTRTOINT':
            return {
                url: `https://llvm.org/docs/LangRef.html#ptrtoint-to-instruction`,
                html: `<html><head></head><body><span id="i-ptrtoint"></span><h4><a class="toc-backref" href="#id2214" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id286">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ptrtoint</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id287">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction converts the pointer or a vector of
pointers <code class="docutils literal notranslate"><span class="pre">value</span></code> to the integer (or vector of integers) type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id288">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction takes a <code class="docutils literal notranslate"><span class="pre">value</span></code> to cast, which must be
a value of type <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> or a vector of pointers, and a
type to cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or
a vector of integers type.</p>
</section>
<section id="id289">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to integer type
<code class="docutils literal notranslate"><span class="pre">ty2</span></code> by interpreting all the pointer representation bits as an integer
(equivalent to a <code class="docutils literal notranslate"><span class="pre">bitcast</span></code>) and either truncating or zero extending that value
to the size of the integer type.
If <code class="docutils literal notranslate"><span class="pre">value</span></code> is smaller than <code class="docutils literal notranslate"><span class="pre">ty2</span></code> then a zero extension is done. If
<code class="docutils literal notranslate"><span class="pre">value</span></code> is larger than <code class="docutils literal notranslate"><span class="pre">ty2</span></code> then a truncation is done. If they are
the same size, then nothing is done (<em>no-op cast</em>) other than a type
change.
The <code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code> always <a class="reference internal" href="#pointercapture"><span class="std std-ref">captures address and provenance</span></a>
of the pointer argument.</p>
</section>
<section id="id290">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">ptrtoint</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%P</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i8</span><span class="w">                         </span><span class="c">; yields truncation on 32-bit architecture</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">ptrtoint</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%P</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i64</span><span class="w">                        </span><span class="c">; yields zero extension on 32-bit architecture</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">ptrtoint</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%P</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i64</span><span class="p">&gt;</span><span class="c">; yields vector zero extension for a vector of addresses on 32-bit architecture</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘ptrtoint’ instruction converts the pointer or a vector ofpointers value to the integer (or vector of integers) type ty2.`,
            };
        case 'PTRTOADDR':
            return {
                url: `https://llvm.org/docs/LangRef.html#ptrtoaddr-to-instruction`,
                html: `<html><head></head><body><span id="i-ptrtoaddr"></span><h4><a class="toc-backref" href="#id2215" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">ptrtoaddr</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id291">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ptrtoaddr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id292">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoaddr</span></code>’ instruction converts the pointer or a vector of
pointers <code class="docutils literal notranslate"><span class="pre">value</span></code> to the underlying integer address (or vector of addresses) of
type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. This is different from <a class="reference internal" href="#i-ptrtoint"><span class="std std-ref">ptrtoint</span></a> in that it
only operates on the index bits of the pointer and ignores all other bits, and
does not capture the provenance of the pointer.</p>
</section>
<section id="id293">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoaddr</span></code>’ instruction takes a <code class="docutils literal notranslate"><span class="pre">value</span></code> to cast, which must be
a value of type <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> or a vector of pointers, and a
type to cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be must be the <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a>
type (or vector of integers) matching the pointer index width of the address
space of <code class="docutils literal notranslate"><span class="pre">ty</span></code>.</p>
</section>
<section id="id294">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoaddr</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to integer type <code class="docutils literal notranslate"><span class="pre">ty2</span></code> by
interpreting the lowest index-width pointer representation bits as an integer.
If the address size and the pointer representation size are the same and
<code class="docutils literal notranslate"><span class="pre">value</span></code> and <code class="docutils literal notranslate"><span class="pre">ty2</span></code> are the same size, then nothing is done (<em>no-op cast</em>)
other than a type change.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">ptrtoaddr</span></code> instruction always <a class="reference internal" href="#pointercapture"><span class="std std-ref">captures the address but not the provenance</span></a>
of the pointer argument.</p>
</section>
<section id="id295">
<h5>Example:</h5>
<p>This example assumes pointers in address space 1 are 64 bits in size with an
address width of 32 bits (<code class="docutils literal notranslate"><span class="pre">p1:64:64:64:32</span></code> <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a>)</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="kt">ptr</span><span class="err">toaddr</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)</span><span class="w"> </span><span class="nv">%P</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">i32</span><span class="w">              </span><span class="c">; extracts low 32 bits of pointer</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="kt">ptr</span><span class="err">toaddr</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)&gt;</span><span class="w"> </span><span class="nv">%P</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w">  </span><span class="c">; yields vector of low 32 bits for each pointer</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘ptrtoaddr’ instruction converts the pointer or a vector ofpointers value to the underlying integer address (or vector of addresses) oftype ty2. This is different from ptrtoint in that itonly operates on the index bits of the pointer and ignores all other bits, anddoes not capture the provenance of the pointer.`,
            };
        case 'INTTOPTR':
            return {
                url: `https://llvm.org/docs/LangRef.html#inttoptr-to-instruction`,
                html: `<html><head></head><body><span id="i-inttoptr"></span><h4><a class="toc-backref" href="#id2216" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id296">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = inttoptr &lt;ty&gt; &lt;value&gt; to &lt;ty2&gt;[, !dereferenceable !&lt;deref_bytes_node&gt;][, !dereferenceable_or_null !&lt;deref_bytes_node&gt;][, !nofree !&lt;empty_node&gt;]            ; yields ty2
</pre></div>
</div>
</section>
<section id="id297">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span></code>’ instruction converts an integer <code class="docutils literal notranslate"><span class="pre">value</span></code> to a
pointer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</section>
<section id="id298">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span></code>’ instruction takes an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> value to
cast, and a type to cast it to, which must be a <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a>
type.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!dereferenceable</span></code> metadata must reference a single metadata
name <code class="docutils literal notranslate"><span class="pre">&lt;deref_bytes_node&gt;</span></code> corresponding to a metadata node with one <code class="docutils literal notranslate"><span class="pre">i64</span></code>
entry.
See <code class="docutils literal notranslate"><span class="pre">dereferenceable</span></code> metadata.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!dereferenceable_or_null</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;deref_bytes_node&gt;</span></code> corresponding to a metadata node with one
<code class="docutils literal notranslate"><span class="pre">i64</span></code> entry.
See <code class="docutils literal notranslate"><span class="pre">dereferenceable_or_null</span></code> metadata.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nofree</span></code> metadata must reference a single metadata name
<code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no entries.
The existence of the <code class="docutils literal notranslate"><span class="pre">!nofree</span></code> metadata on the instruction tells the optimizer
that the memory pointed by the pointer will not be freed after this point.</p>
</section>
<section id="id299">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code> by
applying either a zero extension or a truncation depending on the size
of the integer <code class="docutils literal notranslate"><span class="pre">value</span></code>. If <code class="docutils literal notranslate"><span class="pre">value</span></code> is larger than the size of a
pointer then a truncation is done. If <code class="docutils literal notranslate"><span class="pre">value</span></code> is smaller than the size
of a pointer then a zero extension is done. If they are the same size,
nothing is done (<em>no-op cast</em>).
The behavior is equivalent to a <code class="docutils literal notranslate"><span class="pre">bitcast</span></code>, however, the resulting value is not
guaranteed to be dereferenceable (e.g., if the result type is a
<a class="reference internal" href="#nointptrtype"><span class="std std-ref">non-integral pointers</span></a>).</p>
</section>
<section id="id300">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">inttoptr</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">255</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">ptr</span><span class="w">           </span><span class="c">; yields zero extension on 64-bit architecture</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">inttoptr</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">255</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">ptr</span><span class="w">           </span><span class="c">; yields no-op on 32-bit architecture</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">inttoptr</span><span class="w"> </span><span class="kt">i64</span><span class="w"> </span><span class="m">0</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">ptr</span><span class="w">             </span><span class="c">; yields truncation on 32-bit architecture</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">inttoptr</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">i32</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%G</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="c">; yields truncation of vector G to four pointers</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘inttoptr’ instruction converts an integer value to apointer type, ty2.`,
            };
        case 'BITCAST':
            return {
                url: `https://llvm.org/docs/LangRef.html#bitcast-to-instruction`,
                html: `<html><head></head><body><span id="i-bitcast"></span><h4><a class="toc-backref" href="#id2217" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">bitcast</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id301">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">bitcast</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</section>
<section id="id302">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code> without
changing any bits.</p>
</section>
<section id="id303">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction takes a value to cast, which must be a
non-aggregate first class value, and a type to cast it to, which must
also be a non-aggregate <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. The
bit sizes of <code class="docutils literal notranslate"><span class="pre">value</span></code> and the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, must be
identical. If the source type is a pointer, the destination type must
also be a pointer of the same size. This instruction supports bitwise
conversion of vectors to integers and to vectors of other types (as
long as they have the same size).</p>
</section>
<section id="id304">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. It
is always a <em>no-op cast</em> because no bits change with this
conversion. The conversion is done as if the <code class="docutils literal notranslate"><span class="pre">value</span></code> had been stored
to memory and read back as type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Pointer (or vector of
pointers) types may only be converted to other pointer (or vector of
pointers) types with the same address space through this instruction.
To convert pointers to other types, use the <a class="reference internal" href="#i-inttoptr"><span class="std std-ref">inttoptr</span></a>
or <a class="reference internal" href="#i-ptrtoint"><span class="std std-ref">ptrtoint</span></a> instructions first.</p>
<p>There is a caveat for bitcasts involving vector types in relation to
endianness. For example <code class="docutils literal notranslate"><span class="pre">bitcast</span> <span class="pre">&lt;2</span> <span class="pre">x</span> <span class="pre">i8&gt;</span> <span class="pre">&lt;value&gt;</span> <span class="pre">to</span> <span class="pre">i16</span></code> puts element zero
of the vector in the least significant bits of the i16 for little-endian while
element zero ends up in the most significant bits for big-endian.</p>
</section>
<section id="id305">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>%X = bitcast i8 255 to i8         ; yields i8 :-1
%Y = bitcast i32* %x to i16*      ; yields i16*:%x
%Z = bitcast &lt;2 x i32&gt; %V to i64; ; yields i64: %V (depends on endianness)
%Z = bitcast &lt;2 x i32*&gt; %V to &lt;2 x i64*&gt; ; yields &lt;2 x i64*&gt;
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘bitcast’ instruction converts value to type ty2 withoutchanging any bits.`,
            };
        case 'ADDRSPACECAST':
            return {
                url: `https://llvm.org/docs/LangRef.html#addrspacecast-to-instruction`,
                html: `<html><head></head><body><span id="i-addrspacecast"></span><h4><a class="toc-backref" href="#id2218" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a></h4>
<section id="id306">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">addrspacecast</span> <span class="o">&lt;</span><span class="n">pty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">pty2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">pty2</span>
</pre></div>
</div>
</section>
<section id="id307">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">ptrval</span></code> from <code class="docutils literal notranslate"><span class="pre">pty</span></code> in
address space <code class="docutils literal notranslate"><span class="pre">n</span></code> to type <code class="docutils literal notranslate"><span class="pre">pty2</span></code> in address space <code class="docutils literal notranslate"><span class="pre">m</span></code>.</p>
</section>
<section id="id308">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction takes a pointer or vector of pointer value
to cast and a pointer type to cast it to, which must have a different
address space.</p>
</section>
<section id="id309">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction converts the pointer value
<code class="docutils literal notranslate"><span class="pre">ptrval</span></code> to type <code class="docutils literal notranslate"><span class="pre">pty2</span></code>. It can be a <em>no-op cast</em> or a complex
value modification, depending on the target and the address space
pair. Pointer conversions within the same address space must be
performed with the <code class="docutils literal notranslate"><span class="pre">bitcast</span></code> instruction. Note that if the address
space conversion produces a dereferenceable result then both result
and operand refer to the same memory location. The conversion must
have no side effects, and must not capture the value of the pointer.</p>
<p>If the source is <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a>, the result is
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a>.</p>
<p>If the source is not <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a>, and both source and
destination are <a class="reference internal" href="#nointptrtype"><span class="std std-ref">integral pointers</span></a>, and the
result pointer is dereferenceable, the cast is assumed to be
reversible (i.e., casting the result back to the original address space
should yield the original bit pattern).</p>
<p>Which address space casts are supported depends on the target. Unsupported
address space casts return <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a>.</p>
</section>
<section id="id310">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">addrspacecast</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%x</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">addrspacecast</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)</span><span class="w"> </span><span class="nv">%y</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">2</span><span class="p">)</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">addrspacecast</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">&gt;</span><span class="w"> </span><span class="nv">%z</span><span class="w"> </span><span class="k">to</span><span class="w"> </span><span class="p">&lt;</span><span class="m">4</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="k">addrspace</span><span class="p">(</span><span class="m">3</span><span class="p">)&gt;</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘addrspacecast’ instruction converts ptrval from pty inaddress space n to type pty2 in address space m.`,
            };
        case 'ICMP':
            return {
                url: `https://llvm.org/docs/LangRef.html#icmp-instruction`,
                html: `<html><head></head><body><span id="i-icmp"></span><h4><a class="toc-backref" href="#id2220" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ Instruction</a></h4>
<section id="id311">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">icmp</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">i1</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">icmp</span> <span class="n">samesign</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">i1</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id312">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ instruction returns a boolean value or a vector of
boolean values based on comparison of its two integer, integer vector,
pointer, or pointer vector operands.</p>
</section>
<section id="id313">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ instruction takes three operands. The first operand is
the condition code indicating the kind of comparison to perform. It is
not a value, just a keyword. The possible condition codes are:</p>
<ol class="arabic simple" id="icmp-md-cc">
<li><p><code class="docutils literal notranslate"><span class="pre">eq</span></code>: equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ne</span></code>: not equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: unsigned greater than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uge</span></code>: unsigned greater or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ult</span></code>: unsigned less than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ule</span></code>: unsigned less or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sgt</span></code>: signed greater than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sge</span></code>: signed greater or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">slt</span></code>: signed less than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sle</span></code>: signed less or equal</p></li>
</ol>
<p>The remaining two arguments must be <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or
<a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> or integer <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> typed. They
must also be identical types.</p>
</section>
<section id="id314">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ compares <code class="docutils literal notranslate"><span class="pre">op1</span></code> and <code class="docutils literal notranslate"><span class="pre">op2</span></code> according to the condition
code given as <code class="docutils literal notranslate"><span class="pre">cond</span></code>. The comparison performed always yields either an
<a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a> or vector of <code class="docutils literal notranslate"><span class="pre">i1</span></code> result, as follows:</p>
<ol class="arabic simple" id="icmp-md-cc-sem">
<li><p><code class="docutils literal notranslate"><span class="pre">eq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if the operands are equal, <code class="docutils literal notranslate"><span class="pre">false</span></code>
otherwise. No sign interpretation is necessary or performed.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ne</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if the operands are unequal, <code class="docutils literal notranslate"><span class="pre">false</span></code>
otherwise. No sign interpretation is necessary or performed.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uge</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ult</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ule</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sgt</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sge</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">slt</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">sle</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
</ol>
<p>If the operands are <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> typed, the pointer values
are compared as if they were integers.</p>
<p>If the operands are integer vectors, then they are compared element by
element. The result is an <code class="docutils literal notranslate"><span class="pre">i1</span></code> vector with the same number of elements
as the values being compared. Otherwise, the result is an <code class="docutils literal notranslate"><span class="pre">i1</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">samesign</span></code> keyword is present and the operands are not of the
same sign then the result is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</section>
<section id="id315">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = icmp eq i32 4, 5          ; yields: result=false
&lt;result&gt; = icmp ne ptr %X, %X        ; yields: result=false
&lt;result&gt; = icmp ult i16  4, 5        ; yields: result=true
&lt;result&gt; = icmp sgt i16  4, 5        ; yields: result=false
&lt;result&gt; = icmp ule i16 -4, 5        ; yields: result=false
&lt;result&gt; = icmp sge i16  4, 5        ; yields: result=false
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘icmp’ instruction returns a boolean value or a vector ofboolean values based on comparison of its two integer, integer vector,pointer, or pointer vector operands.`,
            };
        case 'FCMP':
            return {
                url: `https://llvm.org/docs/LangRef.html#fcmp-instruction`,
                html: `<html><head></head><body><span id="i-fcmp"></span><h4><a class="toc-backref" href="#id2221" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ Instruction</a></h4>
<section id="id316">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fcmp</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>     <span class="p">;</span> <span class="n">yields</span> <span class="n">i1</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id317">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction returns a boolean value or vector of boolean
values based on comparison of its operands.</p>
<p>If the operands are floating-point scalars, then the result type is a
boolean (<a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a>).</p>
<p>If the operands are floating-point vectors, then the result type is a
vector of boolean with the same number of elements as the operands being
compared.</p>
</section>
<section id="id318">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction takes three operands. The first operand is
the condition code indicating the kind of comparison to perform. It is
not a value, just a keyword. The possible condition codes are:</p>
<ol class="arabic simple">
<li><p><code class="docutils literal notranslate"><span class="pre">false</span></code>: no comparison, always returns false</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">oeq</span></code>: ordered and equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ogt</span></code>: ordered and greater than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">oge</span></code>: ordered and greater than or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">olt</span></code>: ordered and less than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ole</span></code>: ordered and less than or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">one</span></code>: ordered and not equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ord</span></code>: ordered (no nans)</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ueq</span></code>: unordered or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: unordered or greater than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uge</span></code>: unordered or greater than or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ult</span></code>: unordered or less than</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ule</span></code>: unordered or less than or equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">une</span></code>: unordered or not equal</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uno</span></code>: unordered (either nans)</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">true</span></code>: no comparison, always returns true</p></li>
</ol>
<p><em>Ordered</em> means that neither operand is a QNAN while <em>unordered</em> means
that either operand may be a QNAN.</p>
<p>Each of <code class="docutils literal notranslate"><span class="pre">val1</span></code> and <code class="docutils literal notranslate"><span class="pre">val2</span></code> arguments must be either a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type or a <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of floating-point type.
They must have identical types.</p>
</section>
<section id="id319">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction compares <code class="docutils literal notranslate"><span class="pre">op1</span></code> and <code class="docutils literal notranslate"><span class="pre">op2</span></code> according to the
condition code given as <code class="docutils literal notranslate"><span class="pre">cond</span></code>. If the operands are vectors, then the
vectors are compared element by element. Each comparison performed
always yields an <a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a> result, as follows:</p>
<ol class="arabic simple">
<li><p><code class="docutils literal notranslate"><span class="pre">false</span></code>: always yields <code class="docutils literal notranslate"><span class="pre">false</span></code>, regardless of operands.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">oeq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ogt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">oge</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">olt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ole</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">one</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is not equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ord</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ueq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uge</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ult</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">ule</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">une</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
not equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">uno</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN.</p></li>
<li><p><code class="docutils literal notranslate"><span class="pre">true</span></code>: always yields <code class="docutils literal notranslate"><span class="pre">true</span></code>, regardless of operands.</p></li>
</ol>
<p>The <code class="docutils literal notranslate"><span class="pre">fcmp</span></code> instruction can also optionally take any number of
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>, which are optimization hints to enable
otherwise unsafe floating-point optimizations.</p>
<p>Any set of fast-math flags are legal on an <code class="docutils literal notranslate"><span class="pre">fcmp</span></code> instruction, but the
only flags that have any effect on its semantics are those that allow
assumptions to be made about the values of input arguments; namely
<code class="docutils literal notranslate"><span class="pre">nnan</span></code>, <code class="docutils literal notranslate"><span class="pre">ninf</span></code>, and <code class="docutils literal notranslate"><span class="pre">reassoc</span></code>. See <a class="reference internal" href="#fastmath"><span class="std std-ref">Fast-Math Flags</span></a> for more information.</p>
</section>
<section id="id320">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fcmp oeq float 4.0, 5.0    ; yields: result=false
&lt;result&gt; = fcmp one float 4.0, 5.0    ; yields: result=true
&lt;result&gt; = fcmp olt float 4.0, 5.0    ; yields: result=true
&lt;result&gt; = fcmp ueq double 1.0, 2.0   ; yields: result=false
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘fcmp’ instruction returns a boolean value or vector of booleanvalues based on comparison of its operands.If the operands are floating-point scalars, then the result type is aboolean (i1).If the operands are floating-point vectors, then the result type is avector of boolean with the same number of elements as the operands beingcompared.`,
            };
        case 'PHI':
            return {
                url: `https://llvm.org/docs/LangRef.html#phi-instruction`,
                html: `<html><head></head><body><span id="i-phi"></span><h4><a class="toc-backref" href="#id2222" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ Instruction</a></h4>
<section id="id321">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">phi</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span><span class="o">-</span><span class="n">flags</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="p">[</span> <span class="o">&lt;</span><span class="n">val0</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">label0</span><span class="o">&gt;</span><span class="p">],</span> <span class="o">...</span>
</pre></div>
</div>
</section>
<section id="id322">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction is used to implement the φ node in the SSA
graph representing the function.</p>
</section>
<section id="id323">
<h5>Arguments:</h5>
<p>The type of the incoming values is specified with the first type field.
After this, the ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction takes a list of pairs as
arguments, with one pair for each predecessor basic block of the current
block. Only values of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type may be used as
the value arguments to the PHI node. Only labels may be used as the
label arguments.</p>
<p>There must be no non-phi instructions between the start of a basic block
and the PHI instructions: i.e., PHI instructions must be first in a basic
block.</p>
<p>For the purposes of the SSA form, the use of each incoming value is
deemed to occur on the edge from the corresponding predecessor block to
the current block (but after any definition of an ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’
instruction’s return value on the same edge).</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">fast-math-flags</span></code> marker indicates that the phi has one
or more <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math-flags</span></a>. These are optimization hints
to enable otherwise unsafe floating-point optimizations. Fast-math-flags
are only valid for phis that return <a class="reference internal" href="#fastmath-return-types"><span class="std std-ref">supported floating-point types</span></a>.</p>
</section>
<section id="id324">
<h5>Semantics:</h5>
<p>At runtime, the ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction logically takes on the value
specified by the pair corresponding to the predecessor basic block that
executed just prior to the current block.</p>
</section>
<section id="id325">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">Loop:</span><span class="w">       </span><span class="c">; Infinite loop that counts from 0 on up...</span>
<span class="w">  </span><span class="nv">%indvar</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">phi</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="m">0</span><span class="p">,</span><span class="w"> </span><span class="nv">%LoopHeader</span><span class="w"> </span><span class="p">],</span><span class="w"> </span><span class="p">[</span><span class="w"> </span><span class="nv">%nextindvar</span><span class="p">,</span><span class="w"> </span><span class="nv">%Loop</span><span class="w"> </span><span class="p">]</span>
<span class="w">  </span><span class="nv">%nextindvar</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">add</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="nv">%indvar</span><span class="p">,</span><span class="w"> </span><span class="m">1</span>
<span class="w">  </span><span class="k">br</span><span class="w"> </span><span class="kt">label</span><span class="w"> </span><span class="nv">%Loop</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘phi’ instruction is used to implement the φ node in the SSAgraph representing the function.`,
            };
        case 'SELECT':
            return {
                url: `https://llvm.org/docs/LangRef.html#select-instruction`,
                html: `<html><head></head><body><span id="i-select"></span><h4><a class="toc-backref" href="#id2223" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ Instruction</a></h4>
<section id="id326">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">select</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span> <span class="n">selty</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span>

<span class="n">selty</span> <span class="ow">is</span> <span class="n">either</span> <span class="n">i1</span> <span class="ow">or</span> <span class="p">{</span><span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">}</span>
</pre></div>
</div>
</section>
<section id="id327">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ instruction is used to choose one value based on a
condition, without IR-level branching.</p>
</section>
<section id="id328">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ instruction requires an ‘i1’ value or a vector of ‘i1’
values indicating the condition, and two values of the same <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first
class</span></a> type.</p>
<ol class="arabic simple">
<li><p>The optional <code class="docutils literal notranslate"><span class="pre">fast-math</span> <span class="pre">flags</span></code> marker indicates that the select has one or more
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>. These are optimization hints to enable
otherwise unsafe floating-point optimizations. Fast-math flags are only valid
for selects that return <a class="reference internal" href="#fastmath-return-types"><span class="std std-ref">supported floating-point types</span></a>. Note that the presence of value which would otherwise result
in poison does not cause the result to be poison if the value is on the non-selected arm.
If <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a> are present, they are only applied to the result,
not both arms.</p></li>
</ol>
</section>
<section id="id329">
<h5>Semantics:</h5>
<p>If the condition is an i1 and it evaluates to 1, the instruction returns
the first value argument; otherwise, it returns the second value
argument.</p>
<p>If the condition is a vector of i1, then the value arguments must be
vectors of the same size, and the selection is done element by element.</p>
<p>If the condition is an i1 and the value arguments are vectors of the
same size, then an entire vector is selected.</p>
</section>
<section id="id330">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">select</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="k">true</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">17</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">42</span><span class="w">                   </span><span class="c">; yields i8:17</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">select</span><span class="w"> </span><span class="k">nnan</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="k">true</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">0.0</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="err">NaN</span><span class="w">      </span><span class="c">; yields float:0.0</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">select</span><span class="w"> </span><span class="k">nnan</span><span class="w"> </span><span class="kt">i1</span><span class="w"> </span><span class="k">false</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="m">0.0</span><span class="p">,</span><span class="w"> </span><span class="kt">float</span><span class="w"> </span><span class="err">NaN</span><span class="w">     </span><span class="c">; yields float:poison</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘select’ instruction is used to choose one value based on acondition, without IR-level branching.`,
            };
        case 'FREEZE':
            return {
                url: `https://llvm.org/docs/LangRef.html#freeze-instruction`,
                html: `<html><head></head><body><span id="i-freeze"></span><h4><a class="toc-backref" href="#id2224" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ Instruction</a></h4>
<section id="id331">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">freeze</span> <span class="n">ty</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</section>
<section id="id332">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ instruction is used to stop propagation of
<a class="reference internal" href="#undefvalues"><span class="std std-ref">undef</span></a> and <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a> values.</p>
</section>
<section id="id333">
<h5>Arguments:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ instruction takes a single argument.</p>
</section>
<section id="id334">
<h5>Semantics:</h5>
<p>If the argument is <code class="docutils literal notranslate"><span class="pre">undef</span></code> or <code class="docutils literal notranslate"><span class="pre">poison</span></code>, ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ returns an
arbitrary, but fixed, value of type ‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’.
Otherwise, this instruction is a no-op and returns the input argument.
All uses of a value returned by the same ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ instruction are
guaranteed to always observe the same value, while different ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’
instructions may yield different values.</p>
<p>While <code class="docutils literal notranslate"><span class="pre">undef</span></code> and <code class="docutils literal notranslate"><span class="pre">poison</span></code> pointers can be frozen, the result is a
non-dereferenceable pointer. See the
<a class="reference internal" href="#pointeraliasing"><span class="std std-ref">Pointer Aliasing Rules</span></a> section for more information.
If an aggregate value or vector is frozen, the operand is frozen element-wise.
The padding of an aggregate isn’t considered, since it isn’t visible
without storing it into memory and loading it with a different type.</p>
</section>
<section id="id335">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>%w = i32 undef
%x = freeze i32 %w
%y = add i32 %w, %w         ; undef
%z = add i32 %x, %x         ; even number because all uses of %x observe
                            ; the same value
%x2 = freeze i32 %w
%cmp = icmp eq i32 %x, %x2  ; can be true or false

; example with vectors
%v = &lt;2 x i32&gt; &lt;i32 undef, i32 poison&gt;
%a = extractelement &lt;2 x i32&gt; %v, i32 0    ; undef
%b = extractelement &lt;2 x i32&gt; %v, i32 1    ; poison
%add = add i32 %a, %a                      ; undef

%v.fr = freeze &lt;2 x i32&gt; %v                ; element-wise freeze
%d = extractelement &lt;2 x i32&gt; %v.fr, i32 0 ; not undef
%add.f = add i32 %d, %d                    ; even number

; branching on frozen value
%poison = add nsw i1 %k, undef   ; poison
%c = freeze i1 %poison
br i1 %c, label %foo, label %bar ; non-deterministic branch to %foo or %bar
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘freeze’ instruction is used to stop propagation ofundef and poison values.`,
            };
        case 'CALL':
            return {
                url: `https://llvm.org/docs/LangRef.html#call-instruction`,
                html: `<html><head></head><body><span id="i-call"></span><h4><a class="toc-backref" href="#id2225" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ Instruction</a></h4>
<section id="id336">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="p">[</span><span class="n">tail</span> <span class="o">|</span> <span class="n">musttail</span> <span class="o">|</span> <span class="n">notail</span> <span class="p">]</span> <span class="n">call</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span>
           <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span> <span class="n">operand</span> <span class="n">bundles</span> <span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id337">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ instruction represents a simple function call.</p>
</section>
<section id="id338">
<h5>Arguments:</h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic">
<li><p>The optional <code class="docutils literal notranslate"><span class="pre">tail</span></code> and <code class="docutils literal notranslate"><span class="pre">musttail</span></code> markers indicate that the optimizers
should perform tail call optimization. The <code class="docutils literal notranslate"><span class="pre">tail</span></code> marker is a hint that
<a class="reference external" href="CodeGenerator.html#tail-call-optimization">can be ignored</a>. The
<code class="docutils literal notranslate"><span class="pre">musttail</span></code> marker means that the call must be tail call optimized in order
for the program to be correct. This is true even in the presence of
attributes like “disable-tail-calls”. The <code class="docutils literal notranslate"><span class="pre">musttail</span></code> marker provides these
guarantees:</p>
<ul class="simple">
<li><p>The call will not cause unbounded stack growth if it is part of a
recursive cycle in the call graph.</p></li>
<li><p>Arguments with the <a class="reference internal" href="#attr-inalloca"><span class="std std-ref">inalloca</span></a> or
<a class="reference internal" href="#attr-preallocated"><span class="std std-ref">preallocated</span></a> attribute are forwarded in place.</p></li>
<li><p>If the musttail call appears in a function with the <code class="docutils literal notranslate"><span class="pre">"thunk"</span></code> attribute
and the caller and callee both have varargs, then any unprototyped
arguments in register or memory are forwarded to the callee. Similarly,
the return value of the callee is returned to the caller’s caller, even
if a void return type is in use.</p></li>
</ul>
<p>Both markers imply that the callee does not access allocas, va_args, or
byval arguments from the caller. As an exception to that, an alloca or byval
argument may be passed to the callee as a byval argument, which can be
dereferenced inside the callee. For example:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">declare</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_byval</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="k">byval</span><span class="p">(</span><span class="kt">i64</span><span class="p">))</span>
<span class="k">declare</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_ptr</span><span class="p">(</span><span class="kt">ptr</span><span class="p">)</span>

<span class="c">; Invalid (assuming @take_ptr dereferences the pointer), because %local</span>
<span class="c">; may be de-allocated before the call to @take_ptr.</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@invalid_alloca</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="nv">%local</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i64</span>
<span class="w">  </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_ptr</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%local</span><span class="p">)</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>

<span class="c">; Valid, the byval attribute causes the memory allocated by %local to be</span>
<span class="c">; copied into @take_byval's stack frame.</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@byval_alloca</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="nv">%local</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">alloca</span><span class="w"> </span><span class="kt">i64</span>
<span class="w">  </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_byval</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="k">byval</span><span class="p">(</span><span class="kt">i64</span><span class="p">)</span><span class="w"> </span><span class="nv">%local</span><span class="p">)</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>

<span class="c">; Invalid, because @use_global_va_list uses the variadic arguments from</span>
<span class="c">; @invalid_va_list.</span>
<span class="nv">%struct.va_list</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">type</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="p">}</span>
<span class="vg">@va_list</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">external</span><span class="w"> </span><span class="k">global</span><span class="w"> </span><span class="nv">%struct.va_list</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@use_global_va_list</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="nv">%arg</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">va_arg</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@va_list</span><span class="p">,</span><span class="w"> </span><span class="kt">i64</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@invalid_va_list</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="nv">%a</span><span class="p">,</span><span class="w"> </span><span class="p">...)</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@llvm.va_start.p0</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@va_list</span><span class="p">)</span>
<span class="w">  </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@use_global_va_list</span><span class="p">()</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>

<span class="c">; Valid, byval argument forwarded to tail call as another byval argument.</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@forward_byval</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="k">byval</span><span class="p">(</span><span class="kt">i64</span><span class="p">)</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_byval</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="k">byval</span><span class="p">(</span><span class="kt">i64</span><span class="p">)</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>

<span class="c">; Invalid (assuming @take_ptr dereferences the pointer), byval argument</span>
<span class="c">; passed to tail callee as non-byval ptr.</span>
<span class="k">define</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@invalid_byval</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="k">byval</span><span class="p">(</span><span class="kt">i64</span><span class="p">)</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="nl">entry:</span>
<span class="w">  </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@take_ptr</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%x</span><span class="p">)</span>
<span class="w">  </span><span class="k">ret</span><span class="w"> </span><span class="k">void</span>
<span class="p">}</span>
</pre></div>
</div>
<p>Calls marked <code class="docutils literal notranslate"><span class="pre">musttail</span></code> must obey the following additional rules:</p>
<ul class="simple">
<li><p>The call must immediately precede a <a class="reference internal" href="#i-ret"><span class="std std-ref">ret</span></a> instruction,
or a pointer bitcast followed by a ret instruction.</p></li>
<li><p>The ret instruction must return the (possibly bitcasted) value
produced by the call, undef, or void.</p></li>
<li><p>The calling conventions of the caller and callee must match.</p></li>
<li><p>The callee must be varargs iff the caller is varargs. Bitcasting a
non-varargs function to the appropriate varargs type is legal so
long as the non-varargs prefixes obey the other rules.</p></li>
<li><p>The return type must not undergo automatic conversion to an <cite>sret</cite> pointer.</p></li>
</ul>
<p>In addition, if the calling convention is not <cite>swifttailcc</cite> or <cite>tailcc</cite>:</p>
<ul class="simple">
<li><p>All ABI-impacting function attributes, such as sret, byval, inreg,
returned, and inalloca, must match.</p></li>
<li><p>The caller and callee prototypes must match. Pointer types of parameters
or return types do not differ in address space.</p></li>
</ul>
<p>On the other hand, if the calling convention is <cite>swifttailcc</cite> or <cite>tailcc</cite>:</p>
<ul class="simple">
<li><p>Only these ABI-impacting attributes attributes are allowed: sret, byval,
swiftself, and swiftasync.</p></li>
<li><p>Prototypes are not required to match.</p></li>
</ul>
<p>Tail call optimization for calls marked <code class="docutils literal notranslate"><span class="pre">tail</span></code> is guaranteed to occur if
the following conditions are met:</p>
<ul class="simple">
<li><p>Caller and callee both have the calling convention <code class="docutils literal notranslate"><span class="pre">fastcc</span></code> or <code class="docutils literal notranslate"><span class="pre">tailcc</span></code>.</p></li>
<li><p>The call is in tail position (ret immediately follows call and ret
uses value of call or is void).</p></li>
<li><p>Option <code class="docutils literal notranslate"><span class="pre">-tailcallopt</span></code> is enabled, <code class="docutils literal notranslate"><span class="pre">llvm::GuaranteedTailCallOpt</span></code> is
<code class="docutils literal notranslate"><span class="pre">true</span></code>, or the calling convention is <code class="docutils literal notranslate"><span class="pre">tailcc</span></code>.</p></li>
<li><p><a class="reference external" href="CodeGenerator.html#tail-call-optimization">Platform-specific constraints are met.</a></p></li>
</ul>
</li>
<li><p>The optional <code class="docutils literal notranslate"><span class="pre">notail</span></code> marker indicates that the optimizers should not add
<code class="docutils literal notranslate"><span class="pre">tail</span></code> or <code class="docutils literal notranslate"><span class="pre">musttail</span></code> markers to the call. It is used to prevent tail
call optimization from being performed on the call.</p></li>
<li><p>The optional <code class="docutils literal notranslate"><span class="pre">fast-math</span> <span class="pre">flags</span></code> marker indicates that the call has one or more
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>, which are optimization hints to enable
otherwise unsafe floating-point optimizations. Fast-math flags are only valid
for calls that return <a class="reference internal" href="#fastmath-return-types"><span class="std std-ref">supported floating-point types</span></a>.</p></li>
<li><p>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions. The
calling convention of the call must match the calling convention of
the target function, or else the behavior is undefined.</p></li>
<li><p>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">noext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’
attributes are valid here.</p></li>
<li><p>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>. The signature is computed based on the return type and argument
types.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being called. The
argument types must match the types implied by this signature. This
is only required if the signature specifies a varargs type.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be called. In most cases, this is a direct function call, but
indirect <code class="docutils literal notranslate"><span class="pre">call</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</p></li>
<li><p>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</p></li>
<li><p>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</p></li>
<li><p>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</p></li>
</ol>
</section>
<section id="id339">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ instruction is used to cause control flow to transfer to
a specified function, with its incoming arguments bound to the specified
values. Upon a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction in the called function, control
flow continues with the instruction after the function call, and the
return value of the function is bound to the result argument.</p>
<p>If the callee refers to an intrinsic function, the signature of the call must
match the signature of the callee.  Otherwise, if the signature of the call
does not match the signature of the called function, the behavior is
target-specific.  For a significant mismatch, this likely results in undefined
behavior. LLVM interprocedural optimizations generally only optimize calls
where the signature of the caller matches the signature of the callee.</p>
<p>Note that it is possible for the signatures to mismatch even if a call appears
to be a “direct” call, like <code class="docutils literal notranslate"><span class="pre">call</span> <span class="pre">void</span> <span class="pre">@f()</span></code>.</p>
</section>
<section id="id340">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%retval</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="vg">@test</span><span class="p">(</span><span class="kt">i32</span><span class="w"> </span><span class="nv">%argc</span><span class="p">)</span>
<span class="k">call</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">(</span><span class="kt">ptr</span><span class="p">,</span><span class="w"> </span><span class="p">...)</span><span class="w"> </span><span class="vg">@printf</span><span class="p">(</span><span class="kt">ptr</span><span class="w"> </span><span class="nv">%msg</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="m">12</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="m">42</span><span class="p">)</span><span class="w">        </span><span class="c">; yields i32</span>
<span class="nv">%X</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="vg">@foo</span><span class="p">()</span><span class="w">                                    </span><span class="c">; yields i32</span>
<span class="nv">%Y</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">tail</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">fastcc</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="vg">@foo</span><span class="p">()</span><span class="w">  </span><span class="c">; yields i32</span>
<span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="nv">%foo</span><span class="p">(</span><span class="kt">i8</span><span class="w"> </span><span class="k">signext</span><span class="w"> </span><span class="m">97</span><span class="p">)</span>

<span class="nv">%struct.A</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">type</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">i32</span><span class="p">,</span><span class="w"> </span><span class="kt">i8</span><span class="w"> </span><span class="p">}</span>
<span class="nv">%r</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="nv">%struct.A</span><span class="w"> </span><span class="vg">@foo</span><span class="p">()</span><span class="w">                        </span><span class="c">; yields { i32, i8 }</span>
<span class="nv">%gr</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">extractvalue</span><span class="w"> </span><span class="nv">%struct.A</span><span class="w"> </span><span class="nv">%r</span><span class="p">,</span><span class="w"> </span><span class="m">0</span><span class="w">                </span><span class="c">; yields i32</span>
<span class="nv">%gr1</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">extractvalue</span><span class="w"> </span><span class="nv">%struct.A</span><span class="w"> </span><span class="nv">%r</span><span class="p">,</span><span class="w"> </span><span class="m">1</span><span class="w">               </span><span class="c">; yields i8</span>
<span class="nv">%Z</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">void</span><span class="w"> </span><span class="vg">@foo</span><span class="p">()</span><span class="w"> </span><span class="k">noreturn</span><span class="w">                    </span><span class="c">; indicates that %foo never returns normally</span>
<span class="nv">%ZZ</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">call</span><span class="w"> </span><span class="k">zeroext</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="vg">@bar</span><span class="p">()</span><span class="w">                     </span><span class="c">; Return value is %zero extended</span>
</pre></div>
</div>
<p>llvm treats calls to some functions with names and arguments that match
the standard C99 library as being the C99 library functions, and may
perform optimizations or generate code for them under that assumption.
This is something we’d like to change in the future to provide better
support for freestanding environments and non-C-based languages.</p>
</section>
</body></html>`,
                tooltip: `The ‘call’ instruction represents a simple function call.`,
            };
        case 'VA-ARG':
            return {
                url: `https://llvm.org/docs/LangRef.html#va-arg-instruction`,
                html: `<html><head></head><body><span id="i-va-arg"></span><h4><a class="toc-backref" href="#id2226" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ Instruction</a></h4>
<section id="id341">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">va_arg</span> <span class="o">&lt;</span><span class="n">va_list</span><span class="o">*&gt;</span> <span class="o">&lt;</span><span class="n">arglist</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">argty</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id342">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ instruction is used to access arguments passed through
the “variable argument” area of a function call. It is used to implement
the <code class="docutils literal notranslate"><span class="pre">va_arg</span></code> macro in C.</p>
</section>
<section id="id343">
<h5>Arguments:</h5>
<p>This instruction takes a <code class="docutils literal notranslate"><span class="pre">va_list*</span></code> value and the type of the
argument. It returns a value of the specified argument type and
increments the <code class="docutils literal notranslate"><span class="pre">va_list</span></code> to point to the next argument. The actual
type of <code class="docutils literal notranslate"><span class="pre">va_list</span></code> is target specific.</p>
</section>
<section id="id344">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ instruction loads an argument of the specified type
from the specified <code class="docutils literal notranslate"><span class="pre">va_list</span></code> and causes the <code class="docutils literal notranslate"><span class="pre">va_list</span></code> to point to
the next argument. For more information, see the variable argument
handling <a class="reference internal" href="#int-varargs"><span class="std std-ref">Intrinsic Functions</span></a>.</p>
<p>It is legal for this instruction to be called in a function which does
not take a variable number of arguments, for example, the <code class="docutils literal notranslate"><span class="pre">vfprintf</span></code>
function.</p>
<p><code class="docutils literal notranslate"><span class="pre">va_arg</span></code> is an LLVM instruction instead of an <a class="reference internal" href="#intrinsics"><span class="std std-ref">intrinsic
function</span></a> because it takes a type as an argument.</p>
</section>
<section id="id345">
<h5>Example:</h5>
<p>See the <a class="reference internal" href="#int-varargs"><span class="std std-ref">variable argument processing</span></a> section.</p>
<p>Note that the code generator does not yet fully support va_arg on many
targets. Also, it does not currently support va_arg with aggregate
types on any target.</p>
</section>
</body></html>`,
                tooltip: `The ‘va_arg’ instruction is used to access arguments passed throughthe “variable argument” area of a function call. It is used to implementthe va_arg macro in C.`,
            };
        case 'LANDINGPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#landingpad-instruction`,
                html: `<html><head></head><body><span id="i-landingpad"></span><h4><a class="toc-backref" href="#id2227" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ Instruction</a></h4>
<section id="id346">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">landingpad</span> <span class="o">&lt;</span><span class="n">resultty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;+</span>
<span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">landingpad</span> <span class="o">&lt;</span><span class="n">resultty</span><span class="o">&gt;</span> <span class="n">cleanup</span> <span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;*</span>

<span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;</span> <span class="o">:=</span> <span class="n">catch</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;</span> <span class="o">:=</span> <span class="nb">filter</span> <span class="o">&lt;</span><span class="n">array</span> <span class="n">constant</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">array</span> <span class="n">constant</span><span class="o">&gt;</span>
</pre></div>
</div>
</section>
<section id="id347">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling
system</a> to specify that a basic block
is a landing pad — one where the exception lands, and corresponds to the
code found in the <code class="docutils literal notranslate"><span class="pre">catch</span></code> portion of a <code class="docutils literal notranslate"><span class="pre">try</span></code>/<code class="docutils literal notranslate"><span class="pre">catch</span></code> sequence. It
defines values supplied by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> upon
re-entry to the function. The <code class="docutils literal notranslate"><span class="pre">resultval</span></code> has the type <code class="docutils literal notranslate"><span class="pre">resultty</span></code>.</p>
</section>
<section id="id349">
<h5>Arguments:</h5>
<p>The optional
<code class="docutils literal notranslate"><span class="pre">cleanup</span></code> flag indicates that the landing pad block is a cleanup.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">clause</span></code> begins with the clause type — <code class="docutils literal notranslate"><span class="pre">catch</span></code> or <code class="docutils literal notranslate"><span class="pre">filter</span></code> — and
contains the global variable representing the “type” that may be caught
or filtered respectively. Unlike the <code class="docutils literal notranslate"><span class="pre">catch</span></code> clause, the <code class="docutils literal notranslate"><span class="pre">filter</span></code>
clause takes an array constant as its argument. Use
“<code class="docutils literal notranslate"><span class="pre">[0</span> <span class="pre">x</span> <span class="pre">ptr]</span> <span class="pre">undef</span></code>” for a filter which cannot throw. The
‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction must contain <em>at least</em> one <code class="docutils literal notranslate"><span class="pre">clause</span></code> or
the <code class="docutils literal notranslate"><span class="pre">cleanup</span></code> flag.</p>
</section>
<section id="id350">
<h5>Semantics:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction defines the values which are set by the
<a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> upon re-entry to the function, and
therefore the “result type” of the <code class="docutils literal notranslate"><span class="pre">landingpad</span></code> instruction. As with
calling conventions, how the personality function results are
represented in LLVM IR is target specific.</p>
<p>The clauses are applied in order from top to bottom. If two
<code class="docutils literal notranslate"><span class="pre">landingpad</span></code> instructions are merged together through inlining, the
clauses from the calling function are appended to the list of clauses.
When the call stack is being unwound due to an exception being thrown,
the exception is compared against each <code class="docutils literal notranslate"><span class="pre">clause</span></code> in turn. If it doesn’t
match any of the clauses, and the <code class="docutils literal notranslate"><span class="pre">cleanup</span></code> flag is not set, then
unwinding continues further up the call stack.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">landingpad</span></code> instruction has several restrictions:</p>
<ul class="simple">
<li><p>A landing pad block is a basic block which is the unwind destination
of an ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ instruction.</p></li>
<li><p>A landing pad block must have a ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction as its
first non-PHI instruction.</p></li>
<li><p>There can be only one ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction within the landing
pad block.</p></li>
<li><p>A basic block that is not a landing pad block may not include a
‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction.</p></li>
</ul>
</section>
<section id="id351">
<h5>Example:</h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">;; A landing pad which can catch an integer.</span>
<span class="nv">%res</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">landingpad</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">}</span>
<span class="w">         </span><span class="k">catch</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@_ZTIi</span>
<span class="c">;; A landing pad that is a cleanup.</span>
<span class="nv">%res</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">landingpad</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">}</span>
<span class="w">         </span><span class="k">cleanup</span>
<span class="c">;; A landing pad which can catch an integer and can only throw a double.</span>
<span class="nv">%res</span><span class="w"> </span><span class="p">=</span><span class="w"> </span><span class="k">landingpad</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="kt">ptr</span><span class="p">,</span><span class="w"> </span><span class="kt">i32</span><span class="w"> </span><span class="p">}</span>
<span class="w">         </span><span class="k">catch</span><span class="w"> </span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@_ZTIi</span>
<span class="w">         </span><span class="k">filter</span><span class="w"> </span><span class="p">[</span><span class="m">1</span><span class="w"> </span><span class="k">x</span><span class="w"> </span><span class="kt">ptr</span><span class="p">]</span><span class="w"> </span><span class="p">[</span><span class="kt">ptr</span><span class="w"> </span><span class="vg">@_ZTId</span><span class="p">]</span>
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘landingpad’ instruction is used by LLVM’s exception handlingsystem to specify that a basic blockis a landing pad — one where the exception lands, and corresponds to thecode found in the catch portion of a try/catch sequence. Itdefines values supplied by the personality function uponre-entry to the function. The resultval has the type resultty.`,
            };
        case 'CATCHPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchpad-instruction`,
                html: `<html><head></head><body><span id="i-catchpad"></span><h4><a class="toc-backref" href="#id2228" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">catchpad</span></code>’ Instruction</a></h4>
<section id="id352">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchpad</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">catchswitch</span><span class="o">&gt;</span> <span class="p">[</span><span class="o">&lt;</span><span class="n">args</span><span class="o">&gt;*</span><span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id353">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchpad</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling
system</a> to specify that a basic block
begins a catch handler — one where a personality routine attempts to transfer
control to catch an exception.</p>
</section>
<section id="id355">
<h5>Arguments:</h5>
<p>The <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> operand must always be a token produced by a
<a class="reference internal" href="#i-catchswitch"><span class="std std-ref">catchswitch</span></a> instruction in a predecessor block. This
ensures that each <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> has exactly one predecessor block, and it always
terminates in a <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">args</span></code> correspond to whatever information the personality routine
requires to determine if this is an appropriate handler for the exception. Control
will transfer to the <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> if this is the first appropriate handler for
the exception.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">resultval</span></code> has the type <a class="reference internal" href="#t-token"><span class="std std-ref">token</span></a> and is used to match the
<code class="docutils literal notranslate"><span class="pre">catchpad</span></code> to corresponding <a class="reference internal" href="#i-catchret"><span class="std std-ref">catchrets</span></a> and other nested EH
pads.</p>
</section>
<section id="id356">
<h5>Semantics:</h5>
<p>When the call stack is being unwound due to an exception being thrown, the
exception is compared against the <code class="docutils literal notranslate"><span class="pre">args</span></code>. If it doesn’t match, control will
not reach the <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> instruction.  The representation of <code class="docutils literal notranslate"><span class="pre">args</span></code> is
entirely target and personality function-specific.</p>
<p>Like the <a class="reference internal" href="#i-landingpad"><span class="std std-ref">landingpad</span></a> instruction, the <code class="docutils literal notranslate"><span class="pre">catchpad</span></code>
instruction must be the first non-phi of its parent basic block.</p>
<p>The meaning of the tokens produced and consumed by <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> and other “pad”
instructions is described in the
<a class="reference external" href="ExceptionHandling.html#wineh">Windows exception handling documentation</a>.</p>
<p>When a <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> has been “entered” but not yet “exited” (as
described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
it is undefined behavior to execute a <a class="reference internal" href="#i-call"><span class="std std-ref">call</span></a> or <a class="reference internal" href="#i-invoke"><span class="std std-ref">invoke</span></a>
that does not carry an appropriate <a class="reference internal" href="#ob-funclet"><span class="std std-ref">“funclet” bundle</span></a>.</p>
</section>
<section id="id358">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>dispatch:
  %cs = catchswitch within none [label %handler0] unwind to caller
  ;; A catch block which can catch an integer.
handler0:
  %tok = catchpad within %cs [ptr @_ZTIi]
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘catchpad’ instruction is used by LLVM’s exception handlingsystem to specify that a basic blockbegins a catch handler — one where a personality routine attempts to transfercontrol to catch an exception.`,
            };
        case 'CLEANUPPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#cleanuppad-instruction`,
                html: `<html><head></head><body><span id="i-cleanuppad"></span><h4><a class="toc-backref" href="#id2229" role="doc-backlink">‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ Instruction</a></h4>
<section id="id359">
<h5>Syntax:</h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">cleanuppad</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span><span class="o">&lt;</span><span class="n">args</span><span class="o">&gt;*</span><span class="p">]</span>
</pre></div>
</div>
</section>
<section id="id360">
<h5>Overview:</h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling
system</a> to specify that a basic block
is a cleanup block — one where a personality routine attempts to
transfer control to run cleanup actions.
The <code class="docutils literal notranslate"><span class="pre">args</span></code> correspond to whatever additional
information the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> requires to
execute the cleanup.
The <code class="docutils literal notranslate"><span class="pre">resultval</span></code> has the type <a class="reference internal" href="#t-token"><span class="std std-ref">token</span></a> and is used to
match the <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> to corresponding <a class="reference internal" href="#i-cleanupret"><span class="std std-ref">cleanuprets</span></a>.
The <code class="docutils literal notranslate"><span class="pre">parent</span></code> argument is the token of the funclet that contains the
<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> instruction. If the <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> is not inside a funclet,
this operand may be the token <code class="docutils literal notranslate"><span class="pre">none</span></code>.</p>
</section>
<section id="id362">
<h5>Arguments:</h5>
<p>The instruction takes a list of arbitrary values which are interpreted
by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a>.</p>
</section>
<section id="id363">
<h5>Semantics:</h5>
<p>When the call stack is being unwound due to an exception being thrown,
the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> transfers control to the
<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> with the aid of the personality-specific arguments.
As with calling conventions, how the personality function results are
represented in LLVM IR is target specific.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> instruction has several restrictions:</p>
<ul class="simple">
<li><p>A cleanup block is a basic block which is the unwind destination of
an exceptional instruction.</p></li>
<li><p>A cleanup block must have a ‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction as its
first non-PHI instruction.</p></li>
<li><p>There can be only one ‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction within the
cleanup block.</p></li>
<li><p>A basic block that is not a cleanup block may not include a
‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction.</p></li>
</ul>
<p>When a <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> has been “entered” but not yet “exited” (as
described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
it is undefined behavior to execute a <a class="reference internal" href="#i-call"><span class="std std-ref">call</span></a> or <a class="reference internal" href="#i-invoke"><span class="std std-ref">invoke</span></a>
that does not carry an appropriate <a class="reference internal" href="#ob-funclet"><span class="std std-ref">“funclet” bundle</span></a>.</p>
</section>
<section id="id365">
<h5>Example:</h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>%tok = cleanuppad within %cs []
</pre></div>
</div>
</section>
</body></html>`,
                tooltip: `The ‘cleanuppad’ instruction is used by LLVM’s exception handlingsystem to specify that a basic blockis a cleanup block — one where a personality routine attempts totransfer control to run cleanup actions.The args correspond to whatever additionalinformation the personality function requires toexecute the cleanup.The resultval has the type token and is used tomatch the cleanuppad to corresponding cleanuprets.The parent argument is the token of the funclet that contains thecleanuppad instruction. If the cleanuppad is not inside a funclet,this operand may be the token none.`,
            };
    }
}
