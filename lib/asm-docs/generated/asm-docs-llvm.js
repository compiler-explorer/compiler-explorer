export function getAsmOpcode(opcode) {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case 'RET':
            return {
                url: `https://llvm.org/docs/LangRef.html#ret-instruction`,
                html: `<span id="i-ret"></span><h4><a class="toc-backref" href="#id1748">‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ Instruction</a><a class="headerlink" href="#ret-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="syntax">
<h5><a class="toc-backref" href="#id1749">Syntax:</a><a class="headerlink" href="#syntax" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">ret</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">Return</span> <span class="n">a</span> <span class="n">value</span> <span class="kn">from</span> <span class="nn">a</span> <span class="n">non</span><span class="o">-</span><span class="n">void</span> <span class="n">function</span>
<span class="n">ret</span> <span class="n">void</span>                 <span class="p">;</span> <span class="n">Return</span> <span class="kn">from</span> <span class="nn">void</span> <span class="n">function</span>
</pre></div>
</div>
</div>
<div class="section" id="overview">
<h5><a class="toc-backref" href="#id1750">Overview:</a><a class="headerlink" href="#overview" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction is used to return control flow (and optionally
a value) from a function back to the caller.</p>
<p>There are two forms of the ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction: one that returns a
value and then causes control flow, and one that just causes control
flow to occur.</p>
</div>
<div class="section" id="arguments">
<h5><a class="toc-backref" href="#id1751">Arguments:</a><a class="headerlink" href="#arguments" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction optionally accepts a single argument, the
return value. The type of the return value must be a ‘<a class="reference internal" href="#t-firstclass"><span class="std std-ref">first
class</span></a>’ type.</p>
<p>A function is not <a class="reference internal" href="#wellformed"><span class="std std-ref">well formed</span></a> if it has a non-void
return type and contains a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction with no return value or
a return value with a type that does not match its type, or if it has a
void return type and contains a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction with a return
value.</p>
</div>
<div class="section" id="id29">
<h5><a class="toc-backref" href="#id1752">Semantics:</a><a class="headerlink" href="#id29" title="Permalink to this headline">¶</a></h5>
<p>When the ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction is executed, control flow returns back to
the calling function’s context. If the caller is a
“<a class="reference internal" href="#i-call"><span class="std std-ref">call</span></a>” instruction, execution continues at the
instruction after the call. If the caller was an
“<a class="reference internal" href="#i-invoke"><span class="std std-ref">invoke</span></a>” instruction, execution continues at the
beginning of the “normal” destination block. If the instruction returns
a value, that value shall set the call or invoke instruction’s return
value.</p>
</div>
<div class="section" id="example">
<h5><a class="toc-backref" href="#id1753">Example:</a><a class="headerlink" href="#example" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">ret</span> <span class="kt">i32</span> <span class="m">5</span>                       <span class="c">; Return an integer value of 5</span>
<span class="k">ret</span> <span class="k">void</span>                        <span class="c">; Return from a void function</span>
<span class="k">ret</span> <span class="p">{</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i8</span> <span class="p">}</span> <span class="p">{</span> <span class="kt">i32</span> <span class="m">4</span><span class="p">,</span> <span class="kt">i8</span> <span class="m">2</span> <span class="p">}</span> <span class="c">; Return a struct of values 4 and 2</span>
</pre></div>
</div>
</div>
`,
                tooltip: `There are two forms of the ‘ret’ instruction: one that returns avalue and then causes control flow, and one that just causes control
flow to occur.`,
            };
        case 'BR':
            return {
                url: `https://llvm.org/docs/LangRef.html#br-instruction`,
                html: `<span id="i-br"></span><h4><a class="toc-backref" href="#id1754">‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ Instruction</a><a class="headerlink" href="#br-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id30">
<h5><a class="toc-backref" href="#id1755">Syntax:</a><a class="headerlink" href="#id30" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">br</span> <span class="n">i1</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">iftrue</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">iffalse</span><span class="o">&gt;</span>
<span class="n">br</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">Unconditional</span> <span class="n">branch</span>
</pre></div>
</div>
</div>
<div class="section" id="id31">
<h5><a class="toc-backref" href="#id1756">Overview:</a><a class="headerlink" href="#id31" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction is used to cause control flow to transfer to a
different basic block in the current function. There are two forms of
this instruction, corresponding to a conditional branch and an
unconditional branch.</p>
</div>
<div class="section" id="id32">
<h5><a class="toc-backref" href="#id1757">Arguments:</a><a class="headerlink" href="#id32" title="Permalink to this headline">¶</a></h5>
<p>The conditional branch form of the ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction takes a single
‘<code class="docutils literal notranslate"><span class="pre">i1</span></code>’ value and two ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ values. The unconditional form of the
‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction takes a single ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ value as a target.</p>
</div>
<div class="section" id="id33">
<h5><a class="toc-backref" href="#id1758">Semantics:</a><a class="headerlink" href="#id33" title="Permalink to this headline">¶</a></h5>
<p>Upon execution of a conditional ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’ instruction, the ‘<code class="docutils literal notranslate"><span class="pre">i1</span></code>’
argument is evaluated. If the value is <code class="docutils literal notranslate"><span class="pre">true</span></code>, control flows to the
‘<code class="docutils literal notranslate"><span class="pre">iftrue</span></code>’ <code class="docutils literal notranslate"><span class="pre">label</span></code> argument. If “cond” is <code class="docutils literal notranslate"><span class="pre">false</span></code>, control flows
to the ‘<code class="docutils literal notranslate"><span class="pre">iffalse</span></code>’ <code class="docutils literal notranslate"><span class="pre">label</span></code> argument.
If ‘<code class="docutils literal notranslate"><span class="pre">cond</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</div>
<div class="section" id="id34">
<h5><a class="toc-backref" href="#id1759">Example:</a><a class="headerlink" href="#id34" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">Test:</span>
  <span class="nv">%cond</span> <span class="p">=</span> <span class="k">icmp</span> <span class="k">eq</span> <span class="kt">i32</span> <span class="nv">%a</span><span class="p">,</span> <span class="nv">%b</span>
  <span class="k">br</span> <span class="kt">i1</span> <span class="nv">%cond</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%IfEqual</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%IfUnequal</span>
<span class="nl">IfEqual:</span>
  <span class="k">ret</span> <span class="kt">i32</span> <span class="m">1</span>
<span class="nl">IfUnequal:</span>
  <span class="k">ret</span> <span class="kt">i32</span> <span class="m">0</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The conditional branch form of the ‘br’ instruction takes a single‘i1’ value and two ‘label’ values. The unconditional form of the
‘br’ instruction takes a single ‘label’ value as a target.`,
            };
        case 'SWITCH':
            return {
                url: `https://llvm.org/docs/LangRef.html#switch-instruction`,
                html: `<span id="i-switch"></span><h4><a class="toc-backref" href="#id1760">‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ Instruction</a><a class="headerlink" href="#switch-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id35">
<h5><a class="toc-backref" href="#id1761">Syntax:</a><a class="headerlink" href="#id35" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">switch</span> <span class="o">&lt;</span><span class="n">intty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">defaultdest</span><span class="o">&gt;</span> <span class="p">[</span> <span class="o">&lt;</span><span class="n">intty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest</span><span class="o">&gt;</span> <span class="o">...</span> <span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id36">
<h5><a class="toc-backref" href="#id1762">Overview:</a><a class="headerlink" href="#id36" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction is used to transfer control flow to one of
several different places. It is a generalization of the ‘<code class="docutils literal notranslate"><span class="pre">br</span></code>’
instruction, allowing a branch to occur to one of many possible
destinations.</p>
</div>
<div class="section" id="id37">
<h5><a class="toc-backref" href="#id1763">Arguments:</a><a class="headerlink" href="#id37" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction uses three parameters: an integer
comparison value ‘<code class="docutils literal notranslate"><span class="pre">value</span></code>’, a default ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’ destination, and an
array of pairs of comparison value constants and ‘<code class="docutils literal notranslate"><span class="pre">label</span></code>’s. The table
is not allowed to contain duplicate constant entries.</p>
</div>
<div class="section" id="id38">
<h5><a class="toc-backref" href="#id1764">Semantics:</a><a class="headerlink" href="#id38" title="Permalink to this headline">¶</a></h5>
<p>The <code class="docutils literal notranslate"><span class="pre">switch</span></code> instruction specifies a table of values and destinations.
When the ‘<code class="docutils literal notranslate"><span class="pre">switch</span></code>’ instruction is executed, this table is searched
for the given value. If the value is found, control flow is transferred
to the corresponding destination; otherwise, control flow is transferred
to the default destination.
If ‘<code class="docutils literal notranslate"><span class="pre">value</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</div>
<div class="section" id="implementation">
<h5><a class="toc-backref" href="#id1765">Implementation:</a><a class="headerlink" href="#implementation" title="Permalink to this headline">¶</a></h5>
<p>Depending on properties of the target machine and the particular
<code class="docutils literal notranslate"><span class="pre">switch</span></code> instruction, this instruction may be code generated in
different ways. For example, it could be generated as a series of
chained conditional branches or with a lookup table.</p>
</div>
<div class="section" id="id39">
<h5><a class="toc-backref" href="#id1766">Example:</a><a class="headerlink" href="#id39" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; Emulate a conditional br instruction</span>
<span class="nv">%Val</span> <span class="p">=</span> <span class="k">zext</span> <span class="kt">i1</span> <span class="nv">%value</span> <span class="k">to</span> <span class="kt">i32</span>
<span class="k">switch</span> <span class="kt">i32</span> <span class="nv">%Val</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%truedest</span> <span class="p">[</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%falsedest</span> <span class="p">]</span>

<span class="c">; Emulate an unconditional br instruction</span>
<span class="k">switch</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%dest</span> <span class="p">[</span> <span class="p">]</span>

<span class="c">; Implement a jump table:</span>
<span class="k">switch</span> <span class="kt">i32</span> <span class="nv">%val</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%otherwise</span> <span class="p">[</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%onzero</span>
                                    <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%onone</span>
                                    <span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%ontwo</span> <span class="p">]</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘switch’ instruction uses three parameters: an integercomparison value ‘value’, a default ‘label’ destination, and an
array of pairs of comparison value constants and ‘label’s. The table
is not allowed to contain duplicate constant entries.`,
            };
        case 'INDIRECTBR':
            return {
                url: `https://llvm.org/docs/LangRef.html#indirectbr-instruction`,
                html: `<span id="i-indirectbr"></span><h4><a class="toc-backref" href="#id1767">‘<code class="docutils literal notranslate"><span class="pre">indirectbr</span></code>’ Instruction</a><a class="headerlink" href="#indirectbr-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id40">
<h5><a class="toc-backref" href="#id1768">Syntax:</a><a class="headerlink" href="#id40" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">indirectbr</span> <span class="o">&lt;</span><span class="n">somety</span><span class="o">&gt;*</span> <span class="o">&lt;</span><span class="n">address</span><span class="o">&gt;</span><span class="p">,</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">dest2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id41">
<h5><a class="toc-backref" href="#id1769">Overview:</a><a class="headerlink" href="#id41" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">indirectbr</span></code>’ instruction implements an indirect branch to a
label within the current function, whose address is specified by
“<code class="docutils literal notranslate"><span class="pre">address</span></code>”. Address must be derived from a
<a class="reference internal" href="#blockaddress"><span class="std std-ref">blockaddress</span></a> constant.</p>
</div>
<div class="section" id="id42">
<h5><a class="toc-backref" href="#id1770">Arguments:</a><a class="headerlink" href="#id42" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">address</span></code>’ argument is the address of the label to jump to. The
rest of the arguments indicate the full set of possible destinations
that the address may point to. Blocks are allowed to occur multiple
times in the destination list, though this isn’t particularly useful.</p>
<p>This destination list is required so that dataflow analysis has an
accurate understanding of the CFG.</p>
</div>
<div class="section" id="id43">
<h5><a class="toc-backref" href="#id1771">Semantics:</a><a class="headerlink" href="#id43" title="Permalink to this headline">¶</a></h5>
<p>Control transfers to the block specified in the address argument. All
possible destination blocks must be listed in the label list, otherwise
this instruction has undefined behavior. This implies that jumps to
labels defined in other functions have undefined behavior as well.
If ‘<code class="docutils literal notranslate"><span class="pre">address</span></code>’ is <code class="docutils literal notranslate"><span class="pre">poison</span></code> or <code class="docutils literal notranslate"><span class="pre">undef</span></code>, this instruction has undefined
behavior.</p>
</div>
<div class="section" id="id44">
<h5><a class="toc-backref" href="#id1772">Implementation:</a><a class="headerlink" href="#id44" title="Permalink to this headline">¶</a></h5>
<p>This is typically implemented with a jump through a register.</p>
</div>
<div class="section" id="id45">
<h5><a class="toc-backref" href="#id1773">Example:</a><a class="headerlink" href="#id45" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">indirectbr</span> <span class="kt">i8</span><span class="p">*</span> <span class="nv">%Addr</span><span class="p">,</span> <span class="p">[</span> <span class="kt">label</span> <span class="nv">%bb1</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%bb2</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%bb3</span> <span class="p">]</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘address’ argument is the address of the label to jump to. Therest of the arguments indicate the full set of possible destinations
that the address may point to. Blocks are allowed to occur multiple
times in the destination list, though this isn’t particularly useful.`,
            };
        case 'INVOKE':
            return {
                url: `https://llvm.org/docs/LangRef.html#invoke-instruction`,
                html: `<span id="i-invoke"></span><h4><a class="toc-backref" href="#id1774">‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ Instruction</a><a class="headerlink" href="#invoke-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id46">
<h5><a class="toc-backref" href="#id1775">Syntax:</a><a class="headerlink" href="#id46" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">invoke</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span>
              <span class="p">[</span><span class="n">operand</span> <span class="n">bundles</span><span class="p">]</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">normal</span> <span class="n">label</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">exception</span> <span class="n">label</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id47">
<h5><a class="toc-backref" href="#id1776">Overview:</a><a class="headerlink" href="#id47" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id48">
<h5><a class="toc-backref" href="#id1777">Arguments:</a><a class="headerlink" href="#id48" title="Permalink to this headline">¶</a></h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic simple">
<li>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions.</li>
<li>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’ attributes
are valid here.</li>
<li>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being invoked. The
argument types must match the types implied by this signature. This
type can be omitted if the function is not varargs.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be invoked. In most cases, this is a direct function invocation, but
indirect <code class="docutils literal notranslate"><span class="pre">invoke</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">normal</span> <span class="pre">label</span></code>’: the label reached when the called function
executes a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">exception</span> <span class="pre">label</span></code>’: the label reached when a callee returns via
the <a class="reference internal" href="#i-resume"><span class="std std-ref">resume</span></a> instruction or other exception handling
mechanism.</li>
<li>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</li>
<li>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</li>
</ol>
</div>
<div class="section" id="id49">
<h5><a class="toc-backref" href="#id1778">Semantics:</a><a class="headerlink" href="#id49" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id50">
<h5><a class="toc-backref" href="#id1779">Example:</a><a class="headerlink" href="#id50" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%retval</span> <span class="p">=</span> <span class="k">invoke</span> <span class="kt">i32</span> <span class="vg">@Test</span><span class="p">(</span><span class="kt">i32</span> <span class="m">15</span><span class="p">)</span> <span class="k">to</span> <span class="kt">label</span> <span class="nv">%Continue</span>
            <span class="k">unwind</span> <span class="kt">label</span> <span class="nv">%TestCleanup</span>              <span class="c">; i32:retval set</span>
<span class="nv">%retval</span> <span class="p">=</span> <span class="k">invoke</span> <span class="k">coldcc</span> <span class="kt">i32</span> <span class="nv">%Testfnptr</span><span class="p">(</span><span class="kt">i32</span> <span class="m">15</span><span class="p">)</span> <span class="k">to</span> <span class="kt">label</span> <span class="nv">%Continue</span>
            <span class="k">unwind</span> <span class="kt">label</span> <span class="nv">%TestCleanup</span>              <span class="c">; i32:retval set</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘exception’ label is a landingpad for the exception. As such,
‘exception’ label is required to have the
“landingpad” instruction, which contains the
information about the behavior of the program after unwinding happens,
as its first non-PHI instruction. The restrictions on the
“landingpad” instruction’s tightly couples it to the “invoke”
instruction, so that the important information contained within the
“landingpad” instruction can’t be lost through normal code motion.`,
            };
        case 'CALLBR':
            return {
                url: `https://llvm.org/docs/LangRef.html#callbr-instruction`,
                html: `<span id="i-callbr"></span><h4><a class="toc-backref" href="#id1780">‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ Instruction</a><a class="headerlink" href="#callbr-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id51">
<h5><a class="toc-backref" href="#id1781">Syntax:</a><a class="headerlink" href="#id51" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">callbr</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span>
              <span class="p">[</span><span class="n">operand</span> <span class="n">bundles</span><span class="p">]</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">fallthrough</span> <span class="n">label</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">indirect</span> <span class="n">labels</span><span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id52">
<h5><a class="toc-backref" href="#id1782">Overview:</a><a class="headerlink" href="#id52" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ instruction causes control to transfer to a specified
function, with the possibility of control flow transfer to either the
‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span></code>’ label or one of the ‘<code class="docutils literal notranslate"><span class="pre">indirect</span></code>’ labels.</p>
<p>This instruction should only be used to implement the “goto” feature of gcc
style inline assembly. Any other usage is an error in the IR verifier.</p>
</div>
<div class="section" id="id53">
<h5><a class="toc-backref" href="#id1783">Arguments:</a><a class="headerlink" href="#id53" title="Permalink to this headline">¶</a></h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic simple">
<li>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions.</li>
<li>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’ attributes
are valid here.</li>
<li>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being called. The
argument types must match the types implied by this signature. This
type can be omitted if the function is not varargs.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be called. In most cases, this is a direct function call, but
other <code class="docutils literal notranslate"><span class="pre">callbr</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span> <span class="pre">label</span></code>’: the label reached when the inline assembly’s
execution exits the bottom.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">indirect</span> <span class="pre">labels</span></code>’: the labels reached when a callee transfers control
to a location other than the ‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span> <span class="pre">label</span></code>’. The blockaddress
constant for these should also be in the list of ‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’.</li>
<li>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</li>
<li>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</li>
</ol>
</div>
<div class="section" id="id54">
<h5><a class="toc-backref" href="#id1784">Semantics:</a><a class="headerlink" href="#id54" title="Permalink to this headline">¶</a></h5>
<p>This instruction is designed to operate as a standard ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’
instruction in most regards. The primary difference is that it
establishes an association with additional labels to define where control
flow goes after the call.</p>
<p>The output values of a ‘<code class="docutils literal notranslate"><span class="pre">callbr</span></code>’ instruction are available only to
the ‘<code class="docutils literal notranslate"><span class="pre">fallthrough</span></code>’ block, not to any ‘<code class="docutils literal notranslate"><span class="pre">indirect</span></code>’ blocks(s).</p>
<p>The only use of this today is to implement the “goto” feature of gcc inline
assembly where additional labels can be provided as locations for the inline
assembly to jump to.</p>
</div>
<div class="section" id="id55">
<h5><a class="toc-backref" href="#id1785">Example:</a><a class="headerlink" href="#id55" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span>; "asm goto" without output constraints.
callbr void asm "", "r,X"(i32 %x, i8 *blockaddress(@foo, %indirect))
            to label %fallthrough [label %indirect]

; "asm goto" with output constraints.
&lt;result&gt; = callbr i32 asm "", "=r,r,X"(i32 %x, i8 *blockaddress(@foo, %indirect))
            to label %fallthrough [label %indirect]
</pre></div>
</div>
</div>
`,
                tooltip: `This instruction should only be used to implement the “goto” feature of gccstyle inline assembly. Any other usage is an error in the IR verifier.`,
            };
        case 'RESUME':
            return {
                url: `https://llvm.org/docs/LangRef.html#resume-instruction`,
                html: `<span id="i-resume"></span><h4><a class="toc-backref" href="#id1786">‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ Instruction</a><a class="headerlink" href="#resume-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id56">
<h5><a class="toc-backref" href="#id1787">Syntax:</a><a class="headerlink" href="#id56" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">resume</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id57">
<h5><a class="toc-backref" href="#id1788">Overview:</a><a class="headerlink" href="#id57" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction is a terminator instruction that has no
successors.</p>
</div>
<div class="section" id="id58">
<h5><a class="toc-backref" href="#id1789">Arguments:</a><a class="headerlink" href="#id58" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction requires one argument, which must have the
same type as the result of any ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction in the same
function.</p>
</div>
<div class="section" id="id59">
<h5><a class="toc-backref" href="#id1790">Semantics:</a><a class="headerlink" href="#id59" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">resume</span></code>’ instruction resumes propagation of an existing
(in-flight) exception whose unwinding was interrupted with a
<a class="reference internal" href="#i-landingpad"><span class="std std-ref">landingpad</span></a> instruction.</p>
</div>
<div class="section" id="id60">
<h5><a class="toc-backref" href="#id1791">Example:</a><a class="headerlink" href="#id60" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">resume</span> <span class="p">{</span> <span class="kt">i8</span><span class="p">*,</span> <span class="kt">i32</span> <span class="p">}</span> <span class="nv">%exn</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘resume’ instruction requires one argument, which must have thesame type as the result of any ‘landingpad’ instruction in the same
function.`,
            };
        case 'CATCHSWITCH':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchswitch-instruction`,
                html: `<span id="i-catchswitch"></span><h4><a class="toc-backref" href="#id1792">‘<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>’ Instruction</a><a class="headerlink" href="#catchswitch-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id61">
<h5><a class="toc-backref" href="#id1793">Syntax:</a><a class="headerlink" href="#id61" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchswitch</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span> <span class="n">unwind</span> <span class="n">to</span> <span class="n">caller</span>
<span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchswitch</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler1</span><span class="o">&gt;</span><span class="p">,</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">handler2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">...</span> <span class="p">]</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">default</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id62">
<h5><a class="toc-backref" href="#id1794">Overview:</a><a class="headerlink" href="#id62" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling system</a> to describe the set of possible catch handlers
that may be executed by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">EH personality routine</span></a>.</p>
</div>
<div class="section" id="id63">
<h5><a class="toc-backref" href="#id1795">Arguments:</a><a class="headerlink" href="#id63" title="Permalink to this headline">¶</a></h5>
<p>The <code class="docutils literal notranslate"><span class="pre">parent</span></code> argument is the token of the funclet that contains the
<code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> instruction. If the <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> is not inside a funclet,
this operand may be the token <code class="docutils literal notranslate"><span class="pre">none</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">default</span></code> argument is the label of another basic block beginning with
either a <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> or <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> instruction.  This unwind destination
must be a legal target with respect to the <code class="docutils literal notranslate"><span class="pre">parent</span></code> links, as described in
the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">exception handling documentation</a>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">handlers</span></code> are a nonempty list of successor blocks that each begin with a
<a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a> instruction.</p>
</div>
<div class="section" id="id64">
<h5><a class="toc-backref" href="#id1796">Semantics:</a><a class="headerlink" href="#id64" title="Permalink to this headline">¶</a></h5>
<p>Executing this instruction transfers control to one of the successors in
<code class="docutils literal notranslate"><span class="pre">handlers</span></code>, if appropriate, or continues to unwind via the unwind label if
present.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> is both a terminator and a “pad” instruction, meaning that
it must be both the first non-phi instruction and last instruction in the basic
block. Therefore, it must be the only non-phi instruction in the block.</p>
</div>
<div class="section" id="id65">
<h5><a class="toc-backref" href="#id1797">Example:</a><a class="headerlink" href="#id65" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>dispatch1:
  %cs1 = catchswitch within none [label %handler0, label %handler1] unwind to caller
dispatch2:
  %cs2 = catchswitch within %parenthandler [label %handler0] unwind label %cleanup
</pre></div>
</div>
</div>
`,
                tooltip: `The parent argument is the token of the funclet that contains thecatchswitch instruction. If the catchswitch is not inside a funclet,
this operand may be the token none.`,
            };
        case 'CATCHRET':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchret-instruction`,
                html: `<span id="i-catchret"></span><h4><a class="toc-backref" href="#id1798">‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ Instruction</a><a class="headerlink" href="#catchret-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id66">
<h5><a class="toc-backref" href="#id1799">Syntax:</a><a class="headerlink" href="#id66" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">catchret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">token</span><span class="o">&gt;</span> <span class="n">to</span> <span class="n">label</span> <span class="o">&lt;</span><span class="n">normal</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id67">
<h5><a class="toc-backref" href="#id1800">Overview:</a><a class="headerlink" href="#id67" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ instruction is a terminator instruction that has a
single successor.</p>
</div>
<div class="section" id="id68">
<h5><a class="toc-backref" href="#id1801">Arguments:</a><a class="headerlink" href="#id68" title="Permalink to this headline">¶</a></h5>
<p>The first argument to a ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ indicates which <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> it
exits.  It must be a <a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a>.
The second argument to a ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ specifies where control will
transfer to next.</p>
</div>
<div class="section" id="id69">
<h5><a class="toc-backref" href="#id1802">Semantics:</a><a class="headerlink" href="#id69" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchret</span></code>’ instruction ends an existing (in-flight) exception whose
unwinding was interrupted with a <a class="reference internal" href="#i-catchpad"><span class="std std-ref">catchpad</span></a> instruction.  The
<a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> gets a chance to execute arbitrary
code to, for example, destroy the active exception.  Control then transfers to
<code class="docutils literal notranslate"><span class="pre">normal</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">token</span></code> argument must be a token produced by a <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> instruction.
If the specified <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> is not the most-recently-entered not-yet-exited
funclet pad (as described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
the <code class="docutils literal notranslate"><span class="pre">catchret</span></code>’s behavior is undefined.</p>
</div>
<div class="section" id="id70">
<h5><a class="toc-backref" href="#id1803">Example:</a><a class="headerlink" href="#id70" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>catchret from %catch to label %continue
</pre></div>
</div>
</div>
`,
                tooltip: `The first argument to a ‘catchret’ indicates which catchpad itexits.  It must be a catchpad.
The second argument to a ‘catchret’ specifies where control will
transfer to next.`,
            };
        case 'CLEANUPRET':
            return {
                url: `https://llvm.org/docs/LangRef.html#cleanupret-instruction`,
                html: `<span id="i-cleanupret"></span><h4><a class="toc-backref" href="#id1804">‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ Instruction</a><a class="headerlink" href="#cleanupret-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id71">
<h5><a class="toc-backref" href="#id1805">Syntax:</a><a class="headerlink" href="#id71" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">cleanupret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">label</span> <span class="o">&lt;</span><span class="k">continue</span><span class="o">&gt;</span>
<span class="n">cleanupret</span> <span class="kn">from</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">unwind</span> <span class="n">to</span> <span class="n">caller</span>
</pre></div>
</div>
</div>
<div class="section" id="id72">
<h5><a class="toc-backref" href="#id1806">Overview:</a><a class="headerlink" href="#id72" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction is a terminator instruction that has
an optional successor.</p>
</div>
<div class="section" id="id73">
<h5><a class="toc-backref" href="#id1807">Arguments:</a><a class="headerlink" href="#id73" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id76">
<h5><a class="toc-backref" href="#id1808">Semantics:</a><a class="headerlink" href="#id76" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cleanupret</span></code>’ instruction indicates to the
<a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> that one
<a class="reference internal" href="#i-cleanuppad"><span class="std std-ref">cleanuppad</span></a> it transferred control to has ended.
It transfers control to <code class="docutils literal notranslate"><span class="pre">continue</span></code> or unwinds out of the function.</p>
</div>
<div class="section" id="id77">
<h5><a class="toc-backref" href="#id1809">Example:</a><a class="headerlink" href="#id77" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>cleanupret from %cleanup unwind to caller
cleanupret from %cleanup unwind label %continue
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘cleanupret’ instruction requires one argument, which indicateswhich cleanuppad it exits, and must be a cleanuppad.
If the specified cleanuppad is not the most-recently-entered not-yet-exited
funclet pad (as described in the EH documentation),
the cleanupret’s behavior is undefined.`,
            };
        case 'UNREACHABLE':
            return {
                url: `https://llvm.org/docs/LangRef.html#unreachable-instruction`,
                html: `<span id="i-unreachable"></span><h4><a class="toc-backref" href="#id1810">‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ Instruction</a><a class="headerlink" href="#unreachable-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id78">
<h5><a class="toc-backref" href="#id1811">Syntax:</a><a class="headerlink" href="#id78" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">unreachable</span>
</pre></div>
</div>
</div>
<div class="section" id="id79">
<h5><a class="toc-backref" href="#id1812">Overview:</a><a class="headerlink" href="#id79" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ instruction has no defined semantics. This
instruction is used to inform the optimizer that a particular portion of
the code is not reachable. This can be used to indicate that the code
after a no-return function cannot be reached, and other facts.</p>
</div>
<div class="section" id="id80">
<h5><a class="toc-backref" href="#id1813">Semantics:</a><a class="headerlink" href="#id80" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">unreachable</span></code>’ instruction has no defined semantics.</p>
</div>
`,
                tooltip: `The ‘unreachable’ instruction has no defined semantics.`,
            };
        case 'FNEG':
            return {
                url: `https://llvm.org/docs/LangRef.html#fneg-instruction`,
                html: `<span id="i-fneg"></span><h4><a class="toc-backref" href="#id1815">‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ Instruction</a><a class="headerlink" href="#fneg-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id81">
<h5><a class="toc-backref" href="#id1816">Syntax:</a><a class="headerlink" href="#id81" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fneg</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id82">
<h5><a class="toc-backref" href="#id1817">Overview:</a><a class="headerlink" href="#id82" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ instruction returns the negation of its operand.</p>
</div>
<div class="section" id="id83">
<h5><a class="toc-backref" href="#id1818">Arguments:</a><a class="headerlink" href="#id83" title="Permalink to this headline">¶</a></h5>
<p>The argument to the ‘<code class="docutils literal notranslate"><span class="pre">fneg</span></code>’ instruction must be a
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values.</p>
</div>
<div class="section" id="id84">
<h5><a class="toc-backref" href="#id1819">Semantics:</a><a class="headerlink" href="#id84" title="Permalink to this headline">¶</a></h5>
<p>The value produced is a copy of the operand with its sign bit flipped.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id85">
<h5><a class="toc-backref" href="#id1820">Example:</a><a class="headerlink" href="#id85" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fneg float %val          ; yields float:result = -%var
</pre></div>
</div>
</div>
`,
                tooltip: `The argument to the ‘fneg’ instruction must be afloating-point or vector of
floating-point values.`,
            };
        case 'ADD':
            return {
                url: `https://llvm.org/docs/LangRef.html#add-instruction`,
                html: `<span id="i-add"></span><h4><a class="toc-backref" href="#id1822">‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ Instruction</a><a class="headerlink" href="#add-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id86">
<h5><a class="toc-backref" href="#id1823">Syntax:</a><a class="headerlink" href="#id86" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">add</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id87">
<h5><a class="toc-backref" href="#id1824">Overview:</a><a class="headerlink" href="#id87" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ instruction returns the sum of its two operands.</p>
</div>
<div class="section" id="id88">
<h5><a class="toc-backref" href="#id1825">Arguments:</a><a class="headerlink" href="#id88" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">add</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id89">
<h5><a class="toc-backref" href="#id1826">Semantics:</a><a class="headerlink" href="#id89" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id90">
<h5><a class="toc-backref" href="#id1827">Example:</a><a class="headerlink" href="#id90" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = add i32 4, %var          ; yields i32:result = 4 + %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘add’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'FADD':
            return {
                url: `https://llvm.org/docs/LangRef.html#fadd-instruction`,
                html: `<span id="i-fadd"></span><h4><a class="toc-backref" href="#id1828">‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ Instruction</a><a class="headerlink" href="#fadd-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id91">
<h5><a class="toc-backref" href="#id1829">Syntax:</a><a class="headerlink" href="#id91" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fadd</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id92">
<h5><a class="toc-backref" href="#id1830">Overview:</a><a class="headerlink" href="#id92" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ instruction returns the sum of its two operands.</p>
</div>
<div class="section" id="id93">
<h5><a class="toc-backref" href="#id1831">Arguments:</a><a class="headerlink" href="#id93" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fadd</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</div>
<div class="section" id="id94">
<h5><a class="toc-backref" href="#id1832">Semantics:</a><a class="headerlink" href="#id94" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the floating-point sum of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id95">
<h5><a class="toc-backref" href="#id1833">Example:</a><a class="headerlink" href="#id95" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fadd float 4.0, %var          ; yields float:result = 4.0 + %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘fadd’ instruction must befloating-point or vector of
floating-point values. Both arguments must have identical types.`,
            };
        case 'SUB':
            return {
                url: `https://llvm.org/docs/LangRef.html#sub-instruction`,
                html: `<span id="i-sub"></span><h4><a class="toc-backref" href="#id1834">‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ Instruction</a><a class="headerlink" href="#sub-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id96">
<h5><a class="toc-backref" href="#id1835">Syntax:</a><a class="headerlink" href="#id96" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sub</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id97">
<h5><a class="toc-backref" href="#id1836">Overview:</a><a class="headerlink" href="#id97" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction returns the difference of its two operands.</p>
<p>Note that the ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction is used to represent the ‘<code class="docutils literal notranslate"><span class="pre">neg</span></code>’
instruction present in most other intermediate representations.</p>
</div>
<div class="section" id="id98">
<h5><a class="toc-backref" href="#id1837">Arguments:</a><a class="headerlink" href="#id98" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">sub</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id99">
<h5><a class="toc-backref" href="#id1838">Semantics:</a><a class="headerlink" href="#id99" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id100">
<h5><a class="toc-backref" href="#id1839">Example:</a><a class="headerlink" href="#id100" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = sub i32 4, %var          ; yields i32:result = 4 - %var
&lt;result&gt; = sub i32 0, %val          ; yields i32:result = -%var
</pre></div>
</div>
</div>
`,
                tooltip: `Note that the ‘sub’ instruction is used to represent the ‘neg’instruction present in most other intermediate representations.`,
            };
        case 'FSUB':
            return {
                url: `https://llvm.org/docs/LangRef.html#fsub-instruction`,
                html: `<span id="i-fsub"></span><h4><a class="toc-backref" href="#id1840">‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ Instruction</a><a class="headerlink" href="#fsub-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id101">
<h5><a class="toc-backref" href="#id1841">Syntax:</a><a class="headerlink" href="#id101" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fsub</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id102">
<h5><a class="toc-backref" href="#id1842">Overview:</a><a class="headerlink" href="#id102" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ instruction returns the difference of its two operands.</p>
</div>
<div class="section" id="id103">
<h5><a class="toc-backref" href="#id1843">Arguments:</a><a class="headerlink" href="#id103" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fsub</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</div>
<div class="section" id="id104">
<h5><a class="toc-backref" href="#id1844">Semantics:</a><a class="headerlink" href="#id104" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the floating-point difference of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id105">
<h5><a class="toc-backref" href="#id1845">Example:</a><a class="headerlink" href="#id105" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fsub float 4.0, %var           ; yields float:result = 4.0 - %var
&lt;result&gt; = fsub float -0.0, %val          ; yields float:result = -%var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘fsub’ instruction must befloating-point or vector of
floating-point values. Both arguments must have identical types.`,
            };
        case 'MUL':
            return {
                url: `https://llvm.org/docs/LangRef.html#mul-instruction`,
                html: `<span id="i-mul"></span><h4><a class="toc-backref" href="#id1846">‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ Instruction</a><a class="headerlink" href="#mul-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id106">
<h5><a class="toc-backref" href="#id1847">Syntax:</a><a class="headerlink" href="#id106" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>      <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">mul</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id107">
<h5><a class="toc-backref" href="#id1848">Overview:</a><a class="headerlink" href="#id107" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ instruction returns the product of its two operands.</p>
</div>
<div class="section" id="id108">
<h5><a class="toc-backref" href="#id1849">Arguments:</a><a class="headerlink" href="#id108" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">mul</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id109">
<h5><a class="toc-backref" href="#id1850">Semantics:</a><a class="headerlink" href="#id109" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the integer product of the two operands.</p>
<p>If the result of the multiplication has unsigned overflow, the result
returned is the mathematical result modulo 2<sup>n</sup>, where n is the
bit width of the result.</p>
<p>Because LLVM integers use a two’s complement representation, and the
result is the same width as the operands, this instruction returns the
correct result for both signed and unsigned integers. If a full product
(e.g. <code class="docutils literal notranslate"><span class="pre">i32</span></code> * <code class="docutils literal notranslate"><span class="pre">i32</span></code> -&gt; <code class="docutils literal notranslate"><span class="pre">i64</span></code>) is needed, the operands should be
sign-extended or zero-extended as appropriate to the width of the full
product.</p>
<p><code class="docutils literal notranslate"><span class="pre">nuw</span></code> and <code class="docutils literal notranslate"><span class="pre">nsw</span></code> stand for “No Unsigned Wrap” and “No Signed Wrap”,
respectively. If the <code class="docutils literal notranslate"><span class="pre">nuw</span></code> and/or <code class="docutils literal notranslate"><span class="pre">nsw</span></code> keywords are present, the
result value of the <code class="docutils literal notranslate"><span class="pre">mul</span></code> is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if
unsigned and/or signed overflow, respectively, occurs.</p>
</div>
<div class="section" id="id110">
<h5><a class="toc-backref" href="#id1851">Example:</a><a class="headerlink" href="#id110" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = mul i32 4, %var          ; yields i32:result = 4 * %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘mul’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'FMUL':
            return {
                url: `https://llvm.org/docs/LangRef.html#fmul-instruction`,
                html: `<span id="i-fmul"></span><h4><a class="toc-backref" href="#id1852">‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ Instruction</a><a class="headerlink" href="#fmul-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id111">
<h5><a class="toc-backref" href="#id1853">Syntax:</a><a class="headerlink" href="#id111" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fmul</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id112">
<h5><a class="toc-backref" href="#id1854">Overview:</a><a class="headerlink" href="#id112" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ instruction returns the product of its two operands.</p>
</div>
<div class="section" id="id113">
<h5><a class="toc-backref" href="#id1855">Arguments:</a><a class="headerlink" href="#id113" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fmul</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</div>
<div class="section" id="id114">
<h5><a class="toc-backref" href="#id1856">Semantics:</a><a class="headerlink" href="#id114" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the floating-point product of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id115">
<h5><a class="toc-backref" href="#id1857">Example:</a><a class="headerlink" href="#id115" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fmul float 4.0, %var          ; yields float:result = 4.0 * %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘fmul’ instruction must befloating-point or vector of
floating-point values. Both arguments must have identical types.`,
            };
        case 'UDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#udiv-instruction`,
                html: `<span id="i-udiv"></span><h4><a class="toc-backref" href="#id1858">‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ Instruction</a><a class="headerlink" href="#udiv-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id116">
<h5><a class="toc-backref" href="#id1859">Syntax:</a><a class="headerlink" href="#id116" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">udiv</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">udiv</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id117">
<h5><a class="toc-backref" href="#id1860">Overview:</a><a class="headerlink" href="#id117" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ instruction returns the quotient of its two operands.</p>
</div>
<div class="section" id="id118">
<h5><a class="toc-backref" href="#id1861">Arguments:</a><a class="headerlink" href="#id118" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">udiv</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id119">
<h5><a class="toc-backref" href="#id1862">Semantics:</a><a class="headerlink" href="#id119" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the unsigned integer quotient of the two operands.</p>
<p>Note that unsigned integer division and signed integer division are
distinct operations; for signed integer division, use ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’.</p>
<p>Division by zero is undefined behavior. For vectors, if any element
of the divisor is zero, the operation has undefined behavior.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">udiv</span></code> is
a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if %op1 is not a multiple of %op2 (as
such, “((a udiv exact b) mul b) == a”).</p>
</div>
<div class="section" id="id120">
<h5><a class="toc-backref" href="#id1863">Example:</a><a class="headerlink" href="#id120" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = udiv i32 4, %var          ; yields i32:result = 4 / %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘udiv’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'SDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#sdiv-instruction`,
                html: `<span id="i-sdiv"></span><h4><a class="toc-backref" href="#id1864">‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ Instruction</a><a class="headerlink" href="#sdiv-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id121">
<h5><a class="toc-backref" href="#id1865">Syntax:</a><a class="headerlink" href="#id121" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sdiv</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sdiv</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id122">
<h5><a class="toc-backref" href="#id1866">Overview:</a><a class="headerlink" href="#id122" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ instruction returns the quotient of its two operands.</p>
</div>
<div class="section" id="id123">
<h5><a class="toc-backref" href="#id1867">Arguments:</a><a class="headerlink" href="#id123" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">sdiv</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id124">
<h5><a class="toc-backref" href="#id1868">Semantics:</a><a class="headerlink" href="#id124" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id125">
<h5><a class="toc-backref" href="#id1869">Example:</a><a class="headerlink" href="#id125" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = sdiv i32 4, %var          ; yields i32:result = 4 / %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘sdiv’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'FDIV':
            return {
                url: `https://llvm.org/docs/LangRef.html#fdiv-instruction`,
                html: `<span id="i-fdiv"></span><h4><a class="toc-backref" href="#id1870">‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ Instruction</a><a class="headerlink" href="#fdiv-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id126">
<h5><a class="toc-backref" href="#id1871">Syntax:</a><a class="headerlink" href="#id126" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fdiv</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id127">
<h5><a class="toc-backref" href="#id1872">Overview:</a><a class="headerlink" href="#id127" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ instruction returns the quotient of its two operands.</p>
</div>
<div class="section" id="id128">
<h5><a class="toc-backref" href="#id1873">Arguments:</a><a class="headerlink" href="#id128" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">fdiv</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</div>
<div class="section" id="id129">
<h5><a class="toc-backref" href="#id1874">Semantics:</a><a class="headerlink" href="#id129" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the floating-point quotient of the two operands.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id130">
<h5><a class="toc-backref" href="#id1875">Example:</a><a class="headerlink" href="#id130" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fdiv float 4.0, %var          ; yields float:result = 4.0 / %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘fdiv’ instruction must befloating-point or vector of
floating-point values. Both arguments must have identical types.`,
            };
        case 'UREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#urem-instruction`,
                html: `<span id="i-urem"></span><h4><a class="toc-backref" href="#id1876">‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ Instruction</a><a class="headerlink" href="#urem-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id131">
<h5><a class="toc-backref" href="#id1877">Syntax:</a><a class="headerlink" href="#id131" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">urem</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id132">
<h5><a class="toc-backref" href="#id1878">Overview:</a><a class="headerlink" href="#id132" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ instruction returns the remainder from the unsigned
division of its two arguments.</p>
</div>
<div class="section" id="id133">
<h5><a class="toc-backref" href="#id1879">Arguments:</a><a class="headerlink" href="#id133" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">urem</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id134">
<h5><a class="toc-backref" href="#id1880">Semantics:</a><a class="headerlink" href="#id134" title="Permalink to this headline">¶</a></h5>
<p>This instruction returns the unsigned integer <em>remainder</em> of a division.
This instruction always performs an unsigned division to get the
remainder.</p>
<p>Note that unsigned integer remainder and signed integer remainder are
distinct operations; for signed integer remainder, use ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’.</p>
<p>Taking the remainder of a division by zero is undefined behavior.
For vectors, if any element of the divisor is zero, the operation has
undefined behavior.</p>
</div>
<div class="section" id="id135">
<h5><a class="toc-backref" href="#id1881">Example:</a><a class="headerlink" href="#id135" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = urem i32 4, %var          ; yields i32:result = 4 % %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘urem’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'SREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#srem-instruction`,
                html: `<span id="i-srem"></span><h4><a class="toc-backref" href="#id1882">‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ Instruction</a><a class="headerlink" href="#srem-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id136">
<h5><a class="toc-backref" href="#id1883">Syntax:</a><a class="headerlink" href="#id136" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">srem</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id137">
<h5><a class="toc-backref" href="#id1884">Overview:</a><a class="headerlink" href="#id137" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ instruction returns the remainder from the signed
division of its two operands. This instruction can also take
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> versions of the values in which case the elements
must be integers.</p>
</div>
<div class="section" id="id138">
<h5><a class="toc-backref" href="#id1885">Arguments:</a><a class="headerlink" href="#id138" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">srem</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id139">
<h5><a class="toc-backref" href="#id1886">Semantics:</a><a class="headerlink" href="#id139" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id140">
<h5><a class="toc-backref" href="#id1887">Example:</a><a class="headerlink" href="#id140" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = srem i32 4, %var          ; yields i32:result = 4 % %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘srem’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'FREM':
            return {
                url: `https://llvm.org/docs/LangRef.html#frem-instruction`,
                html: `<span id="i-frem"></span><h4><a class="toc-backref" href="#id1888">‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ Instruction</a><a class="headerlink" href="#frem-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id141">
<h5><a class="toc-backref" href="#id1889">Syntax:</a><a class="headerlink" href="#id141" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">frem</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id142">
<h5><a class="toc-backref" href="#id1890">Overview:</a><a class="headerlink" href="#id142" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ instruction returns the remainder from the division of
its two operands.</p>
</div>
<div class="section" id="id143">
<h5><a class="toc-backref" href="#id1891">Arguments:</a><a class="headerlink" href="#id143" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">frem</span></code>’ instruction must be
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of
floating-point values. Both arguments must have identical types.</p>
</div>
<div class="section" id="id144">
<h5><a class="toc-backref" href="#id1892">Semantics:</a><a class="headerlink" href="#id144" title="Permalink to this headline">¶</a></h5>
<p>The value produced is the floating-point remainder of the two operands.
This is the same output as a libm ‘<code class="docutils literal notranslate"><span class="pre">fmod</span></code>’ function, but without any
possibility of setting <code class="docutils literal notranslate"><span class="pre">errno</span></code>. The remainder has the same sign as the
dividend.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.
This instruction can also take any number of <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math
flags</span></a>, which are optimization hints to enable otherwise
unsafe floating-point optimizations:</p>
</div>
<div class="section" id="id145">
<h5><a class="toc-backref" href="#id1893">Example:</a><a class="headerlink" href="#id145" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = frem float 4.0, %var          ; yields float:result = 4.0 % %var
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘frem’ instruction must befloating-point or vector of
floating-point values. Both arguments must have identical types.`,
            };
        case 'SHL':
            return {
                url: `https://llvm.org/docs/LangRef.html#shl-instruction`,
                html: `<span id="i-shl"></span><h4><a class="toc-backref" href="#id1895">‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ Instruction</a><a class="headerlink" href="#shl-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id146">
<h5><a class="toc-backref" href="#id1896">Syntax:</a><a class="headerlink" href="#id146" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>           <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nuw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shl</span> <span class="n">nuw</span> <span class="n">nsw</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id147">
<h5><a class="toc-backref" href="#id1897">Overview:</a><a class="headerlink" href="#id147" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ instruction returns the first operand shifted to the left
a specified number of bits.</p>
</div>
<div class="section" id="id148">
<h5><a class="toc-backref" href="#id1898">Arguments:</a><a class="headerlink" href="#id148" title="Permalink to this headline">¶</a></h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">shl</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</div>
<div class="section" id="id149">
<h5><a class="toc-backref" href="#id1899">Semantics:</a><a class="headerlink" href="#id149" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id150">
<h5><a class="toc-backref" href="#id1900">Example:</a><a class="headerlink" href="#id150" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = shl i32 4, %var   ; yields i32: 4 &lt;&lt; %var
&lt;result&gt; = shl i32 4, 2      ; yields i32: 16
&lt;result&gt; = shl i32 1, 10     ; yields i32: 1024
&lt;result&gt; = shl i32 1, 32     ; undefined
&lt;result&gt; = shl &lt;2 x i32&gt; &lt; i32 1, i32 1&gt;, &lt; i32 1, i32 2&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 2, i32 4&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `Both arguments to the ‘shl’ instruction must be the sameinteger or vector of integer type.
‘op2’ is treated as an unsigned value.`,
            };
        case 'LSHR':
            return {
                url: `https://llvm.org/docs/LangRef.html#lshr-instruction`,
                html: `<span id="i-lshr"></span><h4><a class="toc-backref" href="#id1901">‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ Instruction</a><a class="headerlink" href="#lshr-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id151">
<h5><a class="toc-backref" href="#id1902">Syntax:</a><a class="headerlink" href="#id151" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">lshr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">lshr</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id152">
<h5><a class="toc-backref" href="#id1903">Overview:</a><a class="headerlink" href="#id152" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ instruction (logical shift right) returns the first
operand shifted to the right a specified number of bits with zero fill.</p>
</div>
<div class="section" id="id153">
<h5><a class="toc-backref" href="#id1904">Arguments:</a><a class="headerlink" href="#id153" title="Permalink to this headline">¶</a></h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">lshr</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</div>
<div class="section" id="id154">
<h5><a class="toc-backref" href="#id1905">Semantics:</a><a class="headerlink" href="#id154" title="Permalink to this headline">¶</a></h5>
<p>This instruction always performs a logical shift right operation. The
most significant bits of the result will be filled with zero bits after
the shift. If <code class="docutils literal notranslate"><span class="pre">op2</span></code> is (statically or dynamically) equal to or larger
than the number of bits in <code class="docutils literal notranslate"><span class="pre">op1</span></code>, this instruction returns a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison
value</span></a>. If the arguments are vectors, each vector element
of <code class="docutils literal notranslate"><span class="pre">op1</span></code> is shifted by the corresponding shift amount in <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">lshr</span></code> is
a poison value if any of the bits shifted out are non-zero.</p>
</div>
<div class="section" id="id155">
<h5><a class="toc-backref" href="#id1906">Example:</a><a class="headerlink" href="#id155" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = lshr i32 4, 1   ; yields i32:result = 2
&lt;result&gt; = lshr i32 4, 2   ; yields i32:result = 1
&lt;result&gt; = lshr i8  4, 3   ; yields i8:result = 0
&lt;result&gt; = lshr i8 -2, 1   ; yields i8:result = 0x7F
&lt;result&gt; = lshr i32 1, 32  ; undefined
&lt;result&gt; = lshr &lt;2 x i32&gt; &lt; i32 -2, i32 4&gt;, &lt; i32 1, i32 2&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 0x7FFFFFFF, i32 1&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `Both arguments to the ‘lshr’ instruction must be the sameinteger or vector of integer type.
‘op2’ is treated as an unsigned value.`,
            };
        case 'ASHR':
            return {
                url: `https://llvm.org/docs/LangRef.html#ashr-instruction`,
                html: `<span id="i-ashr"></span><h4><a class="toc-backref" href="#id1907">‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ Instruction</a><a class="headerlink" href="#ashr-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id156">
<h5><a class="toc-backref" href="#id1908">Syntax:</a><a class="headerlink" href="#id156" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ashr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ashr</span> <span class="n">exact</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id157">
<h5><a class="toc-backref" href="#id1909">Overview:</a><a class="headerlink" href="#id157" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ instruction (arithmetic shift right) returns the first
operand shifted to the right a specified number of bits with sign
extension.</p>
</div>
<div class="section" id="id158">
<h5><a class="toc-backref" href="#id1910">Arguments:</a><a class="headerlink" href="#id158" title="Permalink to this headline">¶</a></h5>
<p>Both arguments to the ‘<code class="docutils literal notranslate"><span class="pre">ashr</span></code>’ instruction must be the same
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer type.
‘<code class="docutils literal notranslate"><span class="pre">op2</span></code>’ is treated as an unsigned value.</p>
</div>
<div class="section" id="id159">
<h5><a class="toc-backref" href="#id1911">Semantics:</a><a class="headerlink" href="#id159" title="Permalink to this headline">¶</a></h5>
<p>This instruction always performs an arithmetic shift right operation,
The most significant bits of the result will be filled with the sign bit
of <code class="docutils literal notranslate"><span class="pre">op1</span></code>. If <code class="docutils literal notranslate"><span class="pre">op2</span></code> is (statically or dynamically) equal to or larger
than the number of bits in <code class="docutils literal notranslate"><span class="pre">op1</span></code>, this instruction returns a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison
value</span></a>. If the arguments are vectors, each vector element
of <code class="docutils literal notranslate"><span class="pre">op1</span></code> is shifted by the corresponding shift amount in <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">exact</span></code> keyword is present, the result value of the <code class="docutils literal notranslate"><span class="pre">ashr</span></code> is
a poison value if any of the bits shifted out are non-zero.</p>
</div>
<div class="section" id="id160">
<h5><a class="toc-backref" href="#id1912">Example:</a><a class="headerlink" href="#id160" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = ashr i32 4, 1   ; yields i32:result = 2
&lt;result&gt; = ashr i32 4, 2   ; yields i32:result = 1
&lt;result&gt; = ashr i8  4, 3   ; yields i8:result = 0
&lt;result&gt; = ashr i8 -2, 1   ; yields i8:result = -1
&lt;result&gt; = ashr i32 1, 32  ; undefined
&lt;result&gt; = ashr &lt;2 x i32&gt; &lt; i32 -2, i32 4&gt;, &lt; i32 1, i32 3&gt;   ; yields: result=&lt;2 x i32&gt; &lt; i32 -1, i32 0&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `Both arguments to the ‘ashr’ instruction must be the sameinteger or vector of integer type.
‘op2’ is treated as an unsigned value.`,
            };
        case 'AND':
            return {
                url: `https://llvm.org/docs/LangRef.html#and-instruction`,
                html: `<span id="i-and"></span><h4><a class="toc-backref" href="#id1913">‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ Instruction</a><a class="headerlink" href="#and-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id161">
<h5><a class="toc-backref" href="#id1914">Syntax:</a><a class="headerlink" href="#id161" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">and</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id162">
<h5><a class="toc-backref" href="#id1915">Overview:</a><a class="headerlink" href="#id162" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction returns the bitwise logical and of its two
operands.</p>
</div>
<div class="section" id="id163">
<h5><a class="toc-backref" href="#id1916">Arguments:</a><a class="headerlink" href="#id163" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id164">
<h5><a class="toc-backref" href="#id1917">Semantics:</a><a class="headerlink" href="#id164" title="Permalink to this headline">¶</a></h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">and</span></code>’ instruction is:</p>
<table border="1" class="docutils">
<colgroup>
<col width="33%">
<col width="33%">
<col width="33%">
</colgroup>
<tbody valign="top">
<tr class="row-odd"><td>In0</td>
<td>In1</td>
<td>Out</td>
</tr>
<tr class="row-even"><td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="row-odd"><td>0</td>
<td>1</td>
<td>0</td>
</tr>
<tr class="row-even"><td>1</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="row-odd"><td>1</td>
<td>1</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>
<div class="section" id="id165">
<h5><a class="toc-backref" href="#id1918">Example:</a><a class="headerlink" href="#id165" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = and i32 4, %var         ; yields i32:result = 4 &amp; %var
&lt;result&gt; = and i32 15, 40          ; yields i32:result = 8
&lt;result&gt; = and i32 4, 8            ; yields i32:result = 0
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘and’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'OR':
            return {
                url: `https://llvm.org/docs/LangRef.html#or-instruction`,
                html: `<span id="i-or"></span><h4><a class="toc-backref" href="#id1919">‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ Instruction</a><a class="headerlink" href="#or-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id166">
<h5><a class="toc-backref" href="#id1920">Syntax:</a><a class="headerlink" href="#id166" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id167">
<h5><a class="toc-backref" href="#id1921">Overview:</a><a class="headerlink" href="#id167" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction returns the bitwise logical inclusive or of its
two operands.</p>
</div>
<div class="section" id="id168">
<h5><a class="toc-backref" href="#id1922">Arguments:</a><a class="headerlink" href="#id168" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id169">
<h5><a class="toc-backref" href="#id1923">Semantics:</a><a class="headerlink" href="#id169" title="Permalink to this headline">¶</a></h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">or</span></code>’ instruction is:</p>
<table border="1" class="docutils">
<colgroup>
<col width="33%">
<col width="33%">
<col width="33%">
</colgroup>
<tbody valign="top">
<tr class="row-odd"><td>In0</td>
<td>In1</td>
<td>Out</td>
</tr>
<tr class="row-even"><td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="row-odd"><td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr class="row-even"><td>1</td>
<td>0</td>
<td>1</td>
</tr>
<tr class="row-odd"><td>1</td>
<td>1</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>
<div class="section" id="id170">
<h5><a class="toc-backref" href="#id1924">Example:</a><a class="headerlink" href="#id170" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">4</span><span class="p">,</span> <span class="o">%</span><span class="n">var</span>         <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">4</span> <span class="o">|</span> <span class="o">%</span><span class="n">var</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">15</span><span class="p">,</span> <span class="mi">40</span>          <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">47</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="ow">or</span> <span class="n">i32</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">8</span>            <span class="p">;</span> <span class="n">yields</span> <span class="n">i32</span><span class="p">:</span><span class="n">result</span> <span class="o">=</span> <span class="mi">12</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘or’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'XOR':
            return {
                url: `https://llvm.org/docs/LangRef.html#xor-instruction`,
                html: `<span id="i-xor"></span><h4><a class="toc-backref" href="#id1925">‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ Instruction</a><a class="headerlink" href="#xor-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id171">
<h5><a class="toc-backref" href="#id1926">Syntax:</a><a class="headerlink" href="#id171" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">xor</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id172">
<h5><a class="toc-backref" href="#id1927">Overview:</a><a class="headerlink" href="#id172" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction returns the bitwise logical exclusive or of
its two operands. The <code class="docutils literal notranslate"><span class="pre">xor</span></code> is used to implement the “one’s
complement” operation, which is the “~” operator in C.</p>
</div>
<div class="section" id="id173">
<h5><a class="toc-backref" href="#id1928">Arguments:</a><a class="headerlink" href="#id173" title="Permalink to this headline">¶</a></h5>
<p>The two arguments to the ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction must be
<a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of integer values. Both
arguments must have identical types.</p>
</div>
<div class="section" id="id174">
<h5><a class="toc-backref" href="#id1929">Semantics:</a><a class="headerlink" href="#id174" title="Permalink to this headline">¶</a></h5>
<p>The truth table used for the ‘<code class="docutils literal notranslate"><span class="pre">xor</span></code>’ instruction is:</p>
<table border="1" class="docutils">
<colgroup>
<col width="33%">
<col width="33%">
<col width="33%">
</colgroup>
<tbody valign="top">
<tr class="row-odd"><td>In0</td>
<td>In1</td>
<td>Out</td>
</tr>
<tr class="row-even"><td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="row-odd"><td>0</td>
<td>1</td>
<td>1</td>
</tr>
<tr class="row-even"><td>1</td>
<td>0</td>
<td>1</td>
</tr>
<tr class="row-odd"><td>1</td>
<td>1</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
<div class="section" id="id175">
<h5><a class="toc-backref" href="#id1930">Example:</a><a class="headerlink" href="#id175" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = xor i32 4, %var         ; yields i32:result = 4 ^ %var
&lt;result&gt; = xor i32 15, 40          ; yields i32:result = 39
&lt;result&gt; = xor i32 4, 8            ; yields i32:result = 12
&lt;result&gt; = xor i32 %V, -1          ; yields i32:result = ~%V
</pre></div>
</div>
</div>
`,
                tooltip: `The two arguments to the ‘xor’ instruction must beinteger or vector of integer values. Both
arguments must have identical types.`,
            };
        case 'EXTRACTELEMENT':
            return {
                url: `https://llvm.org/docs/LangRef.html#extractelement-instruction`,
                html: `<span id="i-extractelement"></span><h4><a class="toc-backref" href="#id1932">‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ Instruction</a><a class="headerlink" href="#extractelement-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id176">
<h5><a class="toc-backref" href="#id1933">Syntax:</a><a class="headerlink" href="#id176" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractelement</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractelement</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id177">
<h5><a class="toc-backref" href="#id1934">Overview:</a><a class="headerlink" href="#id177" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ instruction extracts a single scalar element
from a vector at a specified index.</p>
</div>
<div class="section" id="id178">
<h5><a class="toc-backref" href="#id1935">Arguments:</a><a class="headerlink" href="#id178" title="Permalink to this headline">¶</a></h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">extractelement</span></code>’ instruction is a value of
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> type. The second operand is an index indicating
the position from which to extract the element. The index may be a
variable of any integer type.</p>
</div>
<div class="section" id="id179">
<h5><a class="toc-backref" href="#id1936">Semantics:</a><a class="headerlink" href="#id179" title="Permalink to this headline">¶</a></h5>
<p>The result is a scalar of the same type as the element type of <code class="docutils literal notranslate"><span class="pre">val</span></code>.
Its value is the value at position <code class="docutils literal notranslate"><span class="pre">idx</span></code> of <code class="docutils literal notranslate"><span class="pre">val</span></code>. If <code class="docutils literal notranslate"><span class="pre">idx</span></code>
exceeds the length of <code class="docutils literal notranslate"><span class="pre">val</span></code> for a fixed-length vector, the result is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. For a scalable vector, if the value
of <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the runtime length of the vector, the result is a
<a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</div>
<div class="section" id="id180">
<h5><a class="toc-backref" href="#id1937">Example:</a><a class="headerlink" href="#id180" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = extractelement &lt;4 x i32&gt; %vec, i32 0    ; yields i32
</pre></div>
</div>
</div>
`,
                tooltip: `The first operand of an ‘extractelement’ instruction is a value ofvector type. The second operand is an index indicating
the position from which to extract the element. The index may be a
variable of any integer type.`,
            };
        case 'INSERTELEMENT':
            return {
                url: `https://llvm.org/docs/LangRef.html#insertelement-instruction`,
                html: `<span id="i-insertelement"></span><h4><a class="toc-backref" href="#id1938">‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ Instruction</a><a class="headerlink" href="#insertelement-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id181">
<h5><a class="toc-backref" href="#id1939">Syntax:</a><a class="headerlink" href="#id181" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertelement</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertelement</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span> <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id182">
<h5><a class="toc-backref" href="#id1940">Overview:</a><a class="headerlink" href="#id182" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ instruction inserts a scalar element into a
vector at a specified index.</p>
</div>
<div class="section" id="id183">
<h5><a class="toc-backref" href="#id1941">Arguments:</a><a class="headerlink" href="#id183" title="Permalink to this headline">¶</a></h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">insertelement</span></code>’ instruction is a value of
<a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> type. The second operand is a scalar value whose
type must equal the element type of the first operand. The third operand
is an index indicating the position at which to insert the value. The
index may be a variable of any integer type.</p>
</div>
<div class="section" id="id184">
<h5><a class="toc-backref" href="#id1942">Semantics:</a><a class="headerlink" href="#id184" title="Permalink to this headline">¶</a></h5>
<p>The result is a vector of the same type as <code class="docutils literal notranslate"><span class="pre">val</span></code>. Its element values
are those of <code class="docutils literal notranslate"><span class="pre">val</span></code> except at position <code class="docutils literal notranslate"><span class="pre">idx</span></code>, where it gets the value
<code class="docutils literal notranslate"><span class="pre">elt</span></code>. If <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the length of <code class="docutils literal notranslate"><span class="pre">val</span></code> for a fixed-length vector,
the result is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>. For a scalable vector,
if the value of <code class="docutils literal notranslate"><span class="pre">idx</span></code> exceeds the runtime length of the vector, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</div>
<div class="section" id="id185">
<h5><a class="toc-backref" href="#id1943">Example:</a><a class="headerlink" href="#id185" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = insertelement &lt;4 x i32&gt; %vec, i32 1, i32 0    ; yields &lt;4 x i32&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `The first operand of an ‘insertelement’ instruction is a value ofvector type. The second operand is a scalar value whose
type must equal the element type of the first operand. The third operand
is an index indicating the position at which to insert the value. The
index may be a variable of any integer type.`,
            };
        case 'SHUFFLEVECTOR':
            return {
                url: `https://llvm.org/docs/LangRef.html#shufflevector-instruction`,
                html: `<span id="i-shufflevector"></span><h4><a class="toc-backref" href="#id1944">‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ Instruction</a><a class="headerlink" href="#shufflevector-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id186">
<h5><a class="toc-backref" href="#id1945">Syntax:</a><a class="headerlink" href="#id186" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shufflevector</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v2</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">m</span> <span class="n">x</span> <span class="n">i32</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">mask</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">m</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">shufflevector</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="o">&lt;</span><span class="n">v1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">n</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span> <span class="n">v2</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">m</span> <span class="n">x</span> <span class="n">i32</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">mask</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">vscale</span> <span class="n">x</span> <span class="n">m</span> <span class="n">x</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id187">
<h5><a class="toc-backref" href="#id1946">Overview:</a><a class="headerlink" href="#id187" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ instruction constructs a permutation of elements
from two input vectors, returning a vector with the same element type as
the input and length that is the same as the shuffle mask.</p>
</div>
<div class="section" id="id188">
<h5><a class="toc-backref" href="#id1947">Arguments:</a><a class="headerlink" href="#id188" title="Permalink to this headline">¶</a></h5>
<p>The first two operands of a ‘<code class="docutils literal notranslate"><span class="pre">shufflevector</span></code>’ instruction are vectors
with the same type. The third argument is a shuffle mask vector constant
whose element type is <code class="docutils literal notranslate"><span class="pre">i32</span></code>. The mask vector elements must be constant
integers or <code class="docutils literal notranslate"><span class="pre">undef</span></code> values. The result of the instruction is a vector
whose length is the same as the shuffle mask and whose element type is the
same as the element type of the first two operands.</p>
</div>
<div class="section" id="id189">
<h5><a class="toc-backref" href="#id1948">Semantics:</a><a class="headerlink" href="#id189" title="Permalink to this headline">¶</a></h5>
<p>The elements of the two input vectors are numbered from left to right
across both of the vectors. For each element of the result vector, the
shuffle mask selects an element from one of the input vectors to copy
to the result. Non-negative elements in the mask represent an index
into the concatenated pair of input vectors.</p>
<p>If the shuffle mask is undefined, the result vector is undefined. If
the shuffle mask selects an undefined element from one of the input
vectors, the resulting element is undefined. An undefined element
in the mask vector specifies that the resulting element is undefined.
An undefined element in the mask vector prevents a poisoned vector
element from propagating.</p>
<p>For scalable vectors, the only valid mask values at present are
<code class="docutils literal notranslate"><span class="pre">zeroinitializer</span></code> and <code class="docutils literal notranslate"><span class="pre">undef</span></code>, since we cannot write all indices as
literals for a vector with a length unknown at compile time.</p>
</div>
<div class="section" id="id190">
<h5><a class="toc-backref" href="#id1949">Example:</a><a class="headerlink" href="#id190" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; %v2,
                        &lt;4 x i32&gt; &lt;i32 0, i32 4, i32 1, i32 5&gt;  ; yields &lt;4 x i32&gt;
&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; undef,
                        &lt;4 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3&gt;  ; yields &lt;4 x i32&gt; - Identity shuffle.
&lt;result&gt; = shufflevector &lt;8 x i32&gt; %v1, &lt;8 x i32&gt; undef,
                        &lt;4 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3&gt;  ; yields &lt;4 x i32&gt;
&lt;result&gt; = shufflevector &lt;4 x i32&gt; %v1, &lt;4 x i32&gt; %v2,
                        &lt;8 x i32&gt; &lt;i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 &gt;  ; yields &lt;8 x i32&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `The first two operands of a ‘shufflevector’ instruction are vectorswith the same type. The third argument is a shuffle mask vector constant
whose element type is i32. The mask vector elements must be constant
integers or undef values. The result of the instruction is a vector
whose length is the same as the shuffle mask and whose element type is the
same as the element type of the first two operands.`,
            };
        case 'EXTRACTVALUE':
            return {
                url: `https://llvm.org/docs/LangRef.html#extractvalue-instruction`,
                html: `<span id="i-extractvalue"></span><h4><a class="toc-backref" href="#id1951">‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ Instruction</a><a class="headerlink" href="#extractvalue-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id191">
<h5><a class="toc-backref" href="#id1952">Syntax:</a><a class="headerlink" href="#id191" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">extractvalue</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
</pre></div>
</div>
</div>
<div class="section" id="id192">
<h5><a class="toc-backref" href="#id1953">Overview:</a><a class="headerlink" href="#id192" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction extracts the value of a member field
from an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> value.</p>
</div>
<div class="section" id="id193">
<h5><a class="toc-backref" href="#id1954">Arguments:</a><a class="headerlink" href="#id193" title="Permalink to this headline">¶</a></h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction is a value of
<a class="reference internal" href="#t-struct"><span class="std std-ref">struct</span></a> or <a class="reference internal" href="#t-array"><span class="std std-ref">array</span></a> type. The other operands are
constant indices to specify which value to extract in a similar manner
as indices in a ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction.</p>
<p>The major differences to <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> indexing are:</p>
<ul class="simple">
<li>Since the value being indexed is not a pointer, the first index is
omitted and assumed to be zero.</li>
<li>At least one index must be specified.</li>
<li>Not only struct indices but also array indices must be in bounds.</li>
</ul>
</div>
<div class="section" id="id194">
<h5><a class="toc-backref" href="#id1955">Semantics:</a><a class="headerlink" href="#id194" title="Permalink to this headline">¶</a></h5>
<p>The result is the value at the position in the aggregate specified by
the index operands.</p>
</div>
<div class="section" id="id195">
<h5><a class="toc-backref" href="#id1956">Example:</a><a class="headerlink" href="#id195" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = extractvalue {i32, float} %agg, 0    ; yields i32
</pre></div>
</div>
</div>
`,
                tooltip: `The first operand of an ‘extractvalue’ instruction is a value ofstruct or array type. The other operands are
constant indices to specify which value to extract in a similar manner
as indices in a ‘getelementptr’ instruction.`,
            };
        case 'INSERTVALUE':
            return {
                url: `https://llvm.org/docs/LangRef.html#insertvalue-instruction`,
                html: `<span id="i-insertvalue"></span><h4><a class="toc-backref" href="#id1957">‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ Instruction</a><a class="headerlink" href="#insertvalue-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id196">
<h5><a class="toc-backref" href="#id1958">Syntax:</a><a class="headerlink" href="#id196" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">insertvalue</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">elt</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">{,</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>    <span class="p">;</span> <span class="n">yields</span> <span class="o">&lt;</span><span class="n">aggregate</span> <span class="nb">type</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id197">
<h5><a class="toc-backref" href="#id1959">Overview:</a><a class="headerlink" href="#id197" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ instruction inserts a value into a member field in
an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> value.</p>
</div>
<div class="section" id="id198">
<h5><a class="toc-backref" href="#id1960">Arguments:</a><a class="headerlink" href="#id198" title="Permalink to this headline">¶</a></h5>
<p>The first operand of an ‘<code class="docutils literal notranslate"><span class="pre">insertvalue</span></code>’ instruction is a value of
<a class="reference internal" href="#t-struct"><span class="std std-ref">struct</span></a> or <a class="reference internal" href="#t-array"><span class="std std-ref">array</span></a> type. The second operand is
a first-class value to insert. The following operands are constant
indices indicating the position at which to insert the value in a
similar manner as indices in a ‘<code class="docutils literal notranslate"><span class="pre">extractvalue</span></code>’ instruction. The value
to insert must have the same type as the value identified by the
indices.</p>
</div>
<div class="section" id="id199">
<h5><a class="toc-backref" href="#id1961">Semantics:</a><a class="headerlink" href="#id199" title="Permalink to this headline">¶</a></h5>
<p>The result is an aggregate of the same type as <code class="docutils literal notranslate"><span class="pre">val</span></code>. Its value is
that of <code class="docutils literal notranslate"><span class="pre">val</span></code> except that the value at the position specified by the
indices is that of <code class="docutils literal notranslate"><span class="pre">elt</span></code>.</p>
</div>
<div class="section" id="id200">
<h5><a class="toc-backref" href="#id1962">Example:</a><a class="headerlink" href="#id200" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%agg1</span> <span class="p">=</span> <span class="k">insertvalue</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="kt">float</span><span class="p">}</span> <span class="k">undef</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="m">0</span>              <span class="c">; yields {i32 1, float undef}</span>
<span class="nv">%agg2</span> <span class="p">=</span> <span class="k">insertvalue</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="kt">float</span><span class="p">}</span> <span class="nv">%agg1</span><span class="p">,</span> <span class="kt">float</span> <span class="nv">%val</span><span class="p">,</span> <span class="m">1</span>         <span class="c">; yields {i32 1, float %val}</span>
<span class="nv">%agg3</span> <span class="p">=</span> <span class="k">insertvalue</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="p">{</span><span class="kt">float</span><span class="p">}}</span> <span class="k">undef</span><span class="p">,</span> <span class="kt">float</span> <span class="nv">%val</span><span class="p">,</span> <span class="m">1</span><span class="p">,</span> <span class="m">0</span>    <span class="c">; yields {i32 undef, {float %val}}</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The first operand of an ‘insertvalue’ instruction is a value ofstruct or array type. The second operand is
a first-class value to insert. The following operands are constant
indices indicating the position at which to insert the value in a
similar manner as indices in a ‘extractvalue’ instruction. The value
to insert must have the same type as the value identified by the
indices.`,
            };
        case 'ALLOCA':
            return {
                url: `https://llvm.org/docs/LangRef.html#alloca-instruction`,
                html: `<span id="i-alloca"></span><h4><a class="toc-backref" href="#id1964">‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ Instruction</a><a class="headerlink" href="#alloca-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id201">
<h5><a class="toc-backref" href="#id1965">Syntax:</a><a class="headerlink" href="#id201" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">alloca</span> <span class="p">[</span><span class="n">inalloca</span><span class="p">]</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="p">[,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">NumElements</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">[,</span> <span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span>     <span class="p">;</span> <span class="n">yields</span> <span class="nb">type</span> <span class="n">addrspace</span><span class="p">(</span><span class="n">num</span><span class="p">)</span><span class="o">*</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id202">
<h5><a class="toc-backref" href="#id1966">Overview:</a><a class="headerlink" href="#id202" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ instruction allocates memory on the stack frame of the
currently executing function, to be automatically released when this
function returns to its caller.  If the address space is not explicitly
specified, the object is allocated in the alloca address space from the
<a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a>.</p>
</div>
<div class="section" id="id203">
<h5><a class="toc-backref" href="#id1967">Arguments:</a><a class="headerlink" href="#id203" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">alloca</span></code>’ instruction allocates <code class="docutils literal notranslate"><span class="pre">sizeof(&lt;type&gt;)*NumElements</span></code>
bytes of memory on the runtime stack, returning a pointer of the
appropriate type to the program. If “NumElements” is specified, it is
the number of elements allocated, otherwise “NumElements” is defaulted
to be one. If a constant alignment is specified, the value result of the
allocation is guaranteed to be aligned to at least that boundary. The
alignment may not be greater than <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>. If not specified, or if
zero, the target can choose to align the allocation on any convenient
boundary compatible with the type.</p>
<p>‘<code class="docutils literal notranslate"><span class="pre">type</span></code>’ may be any sized type.</p>
</div>
<div class="section" id="id204">
<h5><a class="toc-backref" href="#id1968">Semantics:</a><a class="headerlink" href="#id204" title="Permalink to this headline">¶</a></h5>
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
target has assigned it a semantics.</p>
<p>If the returned pointer is used by <a class="reference internal" href="#int-lifestart"><span class="std std-ref">llvm.lifetime.start</span></a>,
the returned object is initially dead.
See <a class="reference internal" href="#int-lifestart"><span class="std std-ref">llvm.lifetime.start</span></a> and
<a class="reference internal" href="#int-lifeend"><span class="std std-ref">llvm.lifetime.end</span></a> for the precise semantics of
lifetime-manipulating intrinsics.</p>
</div>
<div class="section" id="id205">
<h5><a class="toc-backref" href="#id1969">Example:</a><a class="headerlink" href="#id205" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span>                             <span class="c">; yields i32*:ptr</span>
<span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">4</span>                      <span class="c">; yields i32*:ptr</span>
<span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">4</span><span class="p">,</span> <span class="k">align</span> <span class="m">1024</span>          <span class="c">; yields i32*:ptr</span>
<span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span><span class="p">,</span> <span class="k">align</span> <span class="m">1024</span>                 <span class="c">; yields i32*:ptr</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘alloca’ instruction allocates sizeof(<type>)*NumElementsbytes of memory on the runtime stack, returning a pointer of the
appropriate type to the program. If “NumElements” is specified, it is
the number of elements allocated, otherwise “NumElements” is defaulted
to be one. If a constant alignment is specified, the value result of the
allocation is guaranteed to be aligned to at least that boundary. The
alignment may not be greater than 1 << 32. If not specified, or if
zero, the target can choose to align the allocation on any convenient
boundary compatible with the type.`,
            };
        case 'LOAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#load-instruction`,
                html: `<span id="i-load"></span><h4><a class="toc-backref" href="#id1970">‘<code class="docutils literal notranslate"><span class="pre">load</span></code>’ Instruction</a><a class="headerlink" href="#load-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id206">
<h5><a class="toc-backref" href="#id1971">Syntax:</a><a class="headerlink" href="#id206" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = load [volatile] &lt;ty&gt;, &lt;ty&gt;* &lt;pointer&gt;[, align &lt;alignment&gt;][, !nontemporal !&lt;nontemp_node&gt;][, !invariant.load !&lt;empty_node&gt;][, !invariant.group !&lt;empty_node&gt;][, !nonnull !&lt;empty_node&gt;][, !dereferenceable !&lt;deref_bytes_node&gt;][, !dereferenceable_or_null !&lt;deref_bytes_node&gt;][, !align !&lt;align_node&gt;][, !noundef !&lt;empty_node&gt;]
&lt;result&gt; = load atomic [volatile] &lt;ty&gt;, &lt;ty&gt;* &lt;pointer&gt; [syncscope("&lt;target-scope&gt;")] &lt;ordering&gt;, align &lt;alignment&gt; [, !invariant.group !&lt;empty_node&gt;]
!&lt;nontemp_node&gt; = !{ i32 1 }
!&lt;empty_node&gt; = !{}
!&lt;deref_bytes_node&gt; = !{ i64 &lt;dereferenceable_bytes&gt; }
!&lt;align_node&gt; = !{ i64 &lt;value_alignment&gt; }
</pre></div>
</div>
</div>
<div class="section" id="id207">
<h5><a class="toc-backref" href="#id1972">Overview:</a><a class="headerlink" href="#id207" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">load</span></code>’ instruction is used to read from memory.</p>
</div>
<div class="section" id="id208">
<h5><a class="toc-backref" href="#id1973">Arguments:</a><a class="headerlink" href="#id208" title="Permalink to this headline">¶</a></h5>
<p>The argument to the <code class="docutils literal notranslate"><span class="pre">load</span></code> instruction specifies the memory address from which
to load. The type specified must be a <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type of
known size (i.e. not containing an <a class="reference internal" href="#t-opaque"><span class="std std-ref">opaque structural type</span></a>). If
the <code class="docutils literal notranslate"><span class="pre">load</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not allowed to
modify the number or order of execution of this <code class="docutils literal notranslate"><span class="pre">load</span></code> with other
<a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">load</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">atomic</span></code>, it takes an extra <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> and optional <code class="docutils literal notranslate"><span class="pre">syncscope("&lt;target-scope&gt;")</span></code> argument. The
<code class="docutils literal notranslate"><span class="pre">release</span></code> and <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code> orderings are not valid on <code class="docutils literal notranslate"><span class="pre">load</span></code> instructions.
Atomic loads produce <a class="reference internal" href="#memmodel"><span class="std std-ref">defined</span></a> results when they may see
multiple atomic stores. The type of the pointee must be an integer, pointer, or
floating-point type whose bit width is a power of two greater than or equal to
eight and less than or equal to a target-specific size limit.  <code class="docutils literal notranslate"><span class="pre">align</span></code> must be
explicitly specified on atomic loads, and the load has undefined behavior if the
alignment is not set to a value which is at least the size in bytes of the
pointee. <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> does not have any defined semantics for atomic loads.</p>
<p>The optional constant <code class="docutils literal notranslate"><span class="pre">align</span></code> argument specifies the alignment of the
operation (that is, the alignment of the memory address). A value of 0
or an omitted <code class="docutils literal notranslate"><span class="pre">align</span></code> argument means that the operation has the ABI
alignment for the target. It is the responsibility of the code emitter
to ensure that the alignment information is correct. Overestimating the
alignment results in undefined behavior. Underestimating the alignment
may produce less efficient code. An alignment of 1 is always safe. The
maximum possible alignment is <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>. An alignment value higher
than the size of the loaded type implies memory up to the alignment
value bytes can be safely loaded without trapping in the default
address space. Access of the high bytes can interfere with debugging
tools, so should not be accessed if the function has the
<code class="docutils literal notranslate"><span class="pre">sanitize_thread</span></code> or <code class="docutils literal notranslate"><span class="pre">sanitize_address</span></code> attributes.</p>
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
<dl class="docutils">
<dt>The optional <code class="docutils literal notranslate"><span class="pre">!invariant.group</span></code> metadata must reference a single metadata name</dt>
<dd><code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no entries.
See <code class="docutils literal notranslate"><span class="pre">invariant.group</span></code> metadata <a class="reference internal" href="#md-invariant-group"><span class="std std-ref">invariant.group</span></a>.</dd>
</dl>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nonnull</span></code> metadata must reference a single
metadata name <code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a metadata node with no
entries. The existence of the <code class="docutils literal notranslate"><span class="pre">!nonnull</span></code> metadata on the
instruction tells the optimizer that the value loaded is known to
never be null. If the value is null at runtime, the behavior is undefined.
This is analogous to the <code class="docutils literal notranslate"><span class="pre">nonnull</span></code> attribute on parameters and return
values. This metadata can only be applied to loads of a pointer type.</p>
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
value is not appropriately aligned at runtime, the behavior is undefined.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!noundef</span></code> metadata must reference a single metadata name
<code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code> corresponding to a node with no entries. The existence of
<code class="docutils literal notranslate"><span class="pre">!noundef</span></code> metadata on the instruction tells the optimizer that the value
loaded is known to be <a class="reference internal" href="#welldefinedvalues"><span class="std std-ref">well defined</span></a>.
If the value isn’t well defined, the behavior is undefined.</p>
</div>
<div class="section" id="id209">
<h5><a class="toc-backref" href="#id1974">Semantics:</a><a class="headerlink" href="#id209" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id210">
<h5><a class="toc-backref" href="#id1975">Examples:</a><a class="headerlink" href="#id210" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span>                               <span class="c">; yields i32*:ptr</span>
<span class="k">store</span> <span class="kt">i32</span> <span class="m">3</span><span class="p">,</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span>                          <span class="c">; yields void</span>
<span class="nv">%val</span> <span class="p">=</span> <span class="k">load</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span>                      <span class="c">; yields i32:val = i32 3</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The argument to the load instruction specifies the memory address from whichto load. The type specified must be a first class type of
known size (i.e. not containing an opaque structural type). If
the load is marked as volatile, then the optimizer is not allowed to
modify the number or order of execution of this load with other
volatile operations.`,
            };
        case 'STORE':
            return {
                url: `https://llvm.org/docs/LangRef.html#store-instruction`,
                html: `<span id="i-store"></span><h4><a class="toc-backref" href="#id1976">‘<code class="docutils literal notranslate"><span class="pre">store</span></code>’ Instruction</a><a class="headerlink" href="#store-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id211">
<h5><a class="toc-backref" href="#id1977">Syntax:</a><a class="headerlink" href="#id211" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>store [volatile] &lt;ty&gt; &lt;value&gt;, &lt;ty&gt;* &lt;pointer&gt;[, align &lt;alignment&gt;][, !nontemporal !&lt;nontemp_node&gt;][, !invariant.group !&lt;empty_node&gt;]        ; yields void
store atomic [volatile] &lt;ty&gt; &lt;value&gt;, &lt;ty&gt;* &lt;pointer&gt; [syncscope("&lt;target-scope&gt;")] &lt;ordering&gt;, align &lt;alignment&gt; [, !invariant.group !&lt;empty_node&gt;] ; yields void
!&lt;nontemp_node&gt; = !{ i32 1 }
!&lt;empty_node&gt; = !{}
</pre></div>
</div>
</div>
<div class="section" id="id212">
<h5><a class="toc-backref" href="#id1978">Overview:</a><a class="headerlink" href="#id212" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">store</span></code>’ instruction is used to write to memory.</p>
</div>
<div class="section" id="id213">
<h5><a class="toc-backref" href="#id1979">Arguments:</a><a class="headerlink" href="#id213" title="Permalink to this headline">¶</a></h5>
<p>There are two arguments to the <code class="docutils literal notranslate"><span class="pre">store</span></code> instruction: a value to store and an
address at which to store it. The type of the <code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code> operand must be a
pointer to the <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type of the <code class="docutils literal notranslate"><span class="pre">&lt;value&gt;</span></code>
operand. If the <code class="docutils literal notranslate"><span class="pre">store</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not
allowed to modify the number or order of execution of this <code class="docutils literal notranslate"><span class="pre">store</span></code> with other
<a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.  Only values of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> types of known size (i.e. not containing an <a class="reference internal" href="#t-opaque"><span class="std std-ref">opaque
structural type</span></a>) can be stored.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">store</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">atomic</span></code>, it takes an extra <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> and optional <code class="docutils literal notranslate"><span class="pre">syncscope("&lt;target-scope&gt;")</span></code> argument. The
<code class="docutils literal notranslate"><span class="pre">acquire</span></code> and <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code> orderings aren’t valid on <code class="docutils literal notranslate"><span class="pre">store</span></code> instructions.
Atomic loads produce <a class="reference internal" href="#memmodel"><span class="std std-ref">defined</span></a> results when they may see
multiple atomic stores. The type of the pointee must be an integer, pointer, or
floating-point type whose bit width is a power of two greater than or equal to
eight and less than or equal to a target-specific size limit.  <code class="docutils literal notranslate"><span class="pre">align</span></code> must be
explicitly specified on atomic stores, and the store has undefined behavior if
the alignment is not set to a value which is at least the size in bytes of the
pointee. <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> does not have any defined semantics for atomic stores.</p>
<p>The optional constant <code class="docutils literal notranslate"><span class="pre">align</span></code> argument specifies the alignment of the
operation (that is, the alignment of the memory address). A value of 0
or an omitted <code class="docutils literal notranslate"><span class="pre">align</span></code> argument means that the operation has the ABI
alignment for the target. It is the responsibility of the code emitter
to ensure that the alignment information is correct. Overestimating the
alignment results in undefined behavior. Underestimating the
alignment may produce less efficient code. An alignment of 1 is always
safe. The maximum possible alignment is <code class="docutils literal notranslate"><span class="pre">1</span> <span class="pre">&lt;&lt;</span> <span class="pre">32</span></code>. An alignment
value higher than the size of the stored type implies memory up to the
alignment value bytes can be stored to without trapping in the default
address space. Storing to the higher bytes however may result in data
races if another thread can access the same address. Introducing a
data race is not allowed. Storing to the extra bytes is not allowed
even in situations where a data race is known to not exist if the
function has the <code class="docutils literal notranslate"><span class="pre">sanitize_address</span></code> attribute.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> metadata must reference a single metadata
name <code class="docutils literal notranslate"><span class="pre">&lt;nontemp_node&gt;</span></code> corresponding to a metadata node with one <code class="docutils literal notranslate"><span class="pre">i32</span></code> entry
of value 1. The existence of the <code class="docutils literal notranslate"><span class="pre">!nontemporal</span></code> metadata on the instruction
tells the optimizer and code generator that this load is not expected to
be reused in the cache. The code generator may select special
instructions to save cache bandwidth, such as the <code class="docutils literal notranslate"><span class="pre">MOVNT</span></code> instruction on
x86.</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">!invariant.group</span></code> metadata must reference a
single metadata name <code class="docutils literal notranslate"><span class="pre">&lt;empty_node&gt;</span></code>. See <code class="docutils literal notranslate"><span class="pre">invariant.group</span></code> metadata.</p>
</div>
<div class="section" id="id214">
<h5><a class="toc-backref" href="#id1980">Semantics:</a><a class="headerlink" href="#id214" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id215">
<h5><a class="toc-backref" href="#id1981">Example:</a><a class="headerlink" href="#id215" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%ptr</span> <span class="p">=</span> <span class="k">alloca</span> <span class="kt">i32</span>                               <span class="c">; yields i32*:ptr</span>
<span class="k">store</span> <span class="kt">i32</span> <span class="m">3</span><span class="p">,</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span>                          <span class="c">; yields void</span>
<span class="nv">%val</span> <span class="p">=</span> <span class="k">load</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span>                      <span class="c">; yields i32:val = i32 3</span>
</pre></div>
</div>
</div>
`,
                tooltip: `There are two arguments to the store instruction: a value to store and anaddress at which to store it. The type of the <pointer> operand must be a
pointer to the first class type of the <value>
operand. If the store is marked as volatile, then the optimizer is not
allowed to modify the number or order of execution of this store with other
volatile operations.  Only values of first class types of known size (i.e. not containing an opaque
structural type) can be stored.`,
            };
        case 'FENCE':
            return {
                url: `https://llvm.org/docs/LangRef.html#fence-instruction`,
                html: `<span id="i-fence"></span><h4><a class="toc-backref" href="#id1982">‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ Instruction</a><a class="headerlink" href="#fence-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id216">
<h5><a class="toc-backref" href="#id1983">Syntax:</a><a class="headerlink" href="#id216" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">fence</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ordering</span><span class="o">&gt;</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">void</span>
</pre></div>
</div>
</div>
<div class="section" id="id217">
<h5><a class="toc-backref" href="#id1984">Overview:</a><a class="headerlink" href="#id217" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ instruction is used to introduce happens-before edges
between operations.</p>
</div>
<div class="section" id="id218">
<h5><a class="toc-backref" href="#id1985">Arguments:</a><a class="headerlink" href="#id218" title="Permalink to this headline">¶</a></h5>
<p>‘<code class="docutils literal notranslate"><span class="pre">fence</span></code>’ instructions take an <a class="reference internal" href="#ordering"><span class="std std-ref">ordering</span></a> argument which
defines what <em>synchronizes-with</em> edges they add. They can only be given
<code class="docutils literal notranslate"><span class="pre">acquire</span></code>, <code class="docutils literal notranslate"><span class="pre">release</span></code>, <code class="docutils literal notranslate"><span class="pre">acq_rel</span></code>, and <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code> orderings.</p>
</div>
<div class="section" id="id219">
<h5><a class="toc-backref" href="#id1986">Semantics:</a><a class="headerlink" href="#id219" title="Permalink to this headline">¶</a></h5>
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
the global program order of other <code class="docutils literal notranslate"><span class="pre">seq_cst</span></code> operations and/or fences.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">fence</span></code> instruction can also take an optional
“<a class="reference internal" href="#syncscope"><span class="std std-ref">syncscope</span></a>” argument.</p>
</div>
<div class="section" id="id220">
<h5><a class="toc-backref" href="#id1987">Example:</a><a class="headerlink" href="#id220" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>fence acquire                                        ; yields void
fence syncscope("singlethread") seq_cst              ; yields void
fence syncscope("agent") seq_cst                     ; yields void
</pre></div>
</div>
</div>
`,
                tooltip: `‘fence’ instructions take an ordering argument whichdefines what synchronizes-with edges they add. They can only be given
acquire, release, acq_rel, and seq_cst orderings.`,
            };
        case 'CMPXCHG':
            return {
                url: `https://llvm.org/docs/LangRef.html#cmpxchg-instruction`,
                html: `<span id="i-cmpxchg"></span><h4><a class="toc-backref" href="#id1988">‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ Instruction</a><a class="headerlink" href="#cmpxchg-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id221">
<h5><a class="toc-backref" href="#id1989">Syntax:</a><a class="headerlink" href="#id221" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">cmpxchg</span> <span class="p">[</span><span class="n">weak</span><span class="p">]</span> <span class="p">[</span><span class="n">volatile</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;*</span> <span class="o">&lt;</span><span class="n">pointer</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">cmp</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">new</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">success</span> <span class="n">ordering</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">failure</span> <span class="n">ordering</span><span class="o">&gt;</span><span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span> <span class="p">;</span> <span class="n">yields</span>  <span class="p">{</span> <span class="n">ty</span><span class="p">,</span> <span class="n">i1</span> <span class="p">}</span>
</pre></div>
</div>
</div>
<div class="section" id="id222">
<h5><a class="toc-backref" href="#id1990">Overview:</a><a class="headerlink" href="#id222" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ instruction is used to atomically modify memory. It
loads a value in memory and compares it to a given value. If they are
equal, it tries to store a new value into the memory.</p>
</div>
<div class="section" id="id223">
<h5><a class="toc-backref" href="#id1991">Arguments:</a><a class="headerlink" href="#id223" title="Permalink to this headline">¶</a></h5>
<p>There are three arguments to the ‘<code class="docutils literal notranslate"><span class="pre">cmpxchg</span></code>’ instruction: an address
to operate on, a value to compare to the value currently be at that
address, and a new value to place at that address if the compared values
are equal. The type of ‘&lt;cmp&gt;’ must be an integer or pointer type whose
bit width is a power of two greater than or equal to eight and less
than or equal to a target-specific size limit. ‘&lt;cmp&gt;’ and ‘&lt;new&gt;’ must
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
<p>The instruction can take an optional <code class="docutils literal notranslate"><span class="pre">align</span></code> attribute.
The alignment must be a power of two greater or equal to the size of the
<cite>&lt;value&gt;</cite> type. If unspecified, the alignment is assumed to be equal to the
size of the ‘&lt;value&gt;’ type. Note that this default alignment assumption is
different from the alignment used for the load/store instructions when align
isn’t specified.</p>
<p>The pointer passed into cmpxchg must have alignment greater than or
equal to the size in memory of the operand.</p>
</div>
<div class="section" id="id224">
<h5><a class="toc-backref" href="#id1992">Semantics:</a><a class="headerlink" href="#id224" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id225">
<h5><a class="toc-backref" href="#id1993">Example:</a><a class="headerlink" href="#id225" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">entry:</span>
  <span class="nv">%orig</span> <span class="p">=</span> <span class="k">load</span> <span class="k">atomic</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span> <span class="k">unordered</span><span class="p">,</span> <span class="k">align</span> <span class="m">4</span>                      <span class="c">; yields i32</span>
  <span class="k">br</span> <span class="kt">label</span> <span class="nv">%loop</span>

<span class="nl">loop:</span>
  <span class="nv">%cmp</span> <span class="p">=</span> <span class="k">phi</span> <span class="kt">i32</span> <span class="p">[</span> <span class="nv">%orig</span><span class="p">,</span> <span class="nv">%entry</span> <span class="p">],</span> <span class="p">[</span><span class="nv">%value_loaded</span><span class="p">,</span> <span class="nv">%loop</span><span class="p">]</span>
  <span class="nv">%squared</span> <span class="p">=</span> <span class="k">mul</span> <span class="kt">i32</span> <span class="nv">%cmp</span><span class="p">,</span> <span class="nv">%cmp</span>
  <span class="nv">%val_success</span> <span class="p">=</span> <span class="k">cmpxchg</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span><span class="p">,</span> <span class="kt">i32</span> <span class="nv">%cmp</span><span class="p">,</span> <span class="kt">i32</span> <span class="nv">%squared</span> <span class="k">acq_rel</span> <span class="k">monotonic</span> <span class="c">; yields  { i32, i1 }</span>
  <span class="nv">%value_loaded</span> <span class="p">=</span> <span class="k">extractvalue</span> <span class="p">{</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i1</span> <span class="p">}</span> <span class="nv">%val_success</span><span class="p">,</span> <span class="m">0</span>
  <span class="nv">%success</span> <span class="p">=</span> <span class="k">extractvalue</span> <span class="p">{</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i1</span> <span class="p">}</span> <span class="nv">%val_success</span><span class="p">,</span> <span class="m">1</span>
  <span class="k">br</span> <span class="kt">i1</span> <span class="nv">%success</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%done</span><span class="p">,</span> <span class="kt">label</span> <span class="nv">%loop</span>

<span class="nl">done:</span>
  <span class="p">...</span>
</pre></div>
</div>
</div>
`,
                tooltip: `There are three arguments to the ‘cmpxchg’ instruction: an addressto operate on, a value to compare to the value currently be at that
address, and a new value to place at that address if the compared values
are equal. The type of ‘<cmp>’ must be an integer or pointer type whose
bit width is a power of two greater than or equal to eight and less
than or equal to a target-specific size limit. ‘<cmp>’ and ‘<new>’ must
have the same type, and the type of ‘<pointer>’ must be a pointer to
that type. If the cmpxchg is marked as volatile, then the
optimizer is not allowed to modify the number or order of execution of
this cmpxchg with other volatile operations.`,
            };
        case 'ATOMICRMW':
            return {
                url: `https://llvm.org/docs/LangRef.html#atomicrmw-instruction`,
                html: `<span id="i-atomicrmw"></span><h4><a class="toc-backref" href="#id1994">‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ Instruction</a><a class="headerlink" href="#atomicrmw-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id226">
<h5><a class="toc-backref" href="#id1995">Syntax:</a><a class="headerlink" href="#id226" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">atomicrmw</span> <span class="p">[</span><span class="n">volatile</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">operation</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;*</span> <span class="o">&lt;</span><span class="n">pointer</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="p">[</span><span class="n">syncscope</span><span class="p">(</span><span class="s2">"&lt;target-scope&gt;"</span><span class="p">)]</span> <span class="o">&lt;</span><span class="n">ordering</span><span class="o">&gt;</span><span class="p">[,</span> <span class="n">align</span> <span class="o">&lt;</span><span class="n">alignment</span><span class="o">&gt;</span><span class="p">]</span>  <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span>
</pre></div>
</div>
</div>
<div class="section" id="id227">
<h5><a class="toc-backref" href="#id1996">Overview:</a><a class="headerlink" href="#id227" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ instruction is used to atomically modify memory.</p>
</div>
<div class="section" id="id228">
<h5><a class="toc-backref" href="#id1997">Arguments:</a><a class="headerlink" href="#id228" title="Permalink to this headline">¶</a></h5>
<p>There are three arguments to the ‘<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code>’ instruction: an
operation to apply, an address whose value to modify, an argument to the
operation. The operation must be one of the following keywords:</p>
<ul class="simple">
<li>xchg</li>
<li>add</li>
<li>sub</li>
<li>and</li>
<li>nand</li>
<li>or</li>
<li>xor</li>
<li>max</li>
<li>min</li>
<li>umax</li>
<li>umin</li>
<li>fadd</li>
<li>fsub</li>
</ul>
<p>For most of these operations, the type of ‘&lt;value&gt;’ must be an integer
type whose bit width is a power of two greater than or equal to eight
and less than or equal to a target-specific size limit. For xchg, this
may also be a floating point or a pointer type with the same size constraints
as integers.  For fadd/fsub, this must be a floating point type.  The
type of the ‘<code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code>’ operand must be a pointer to that type. If
the <code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> is marked as <code class="docutils literal notranslate"><span class="pre">volatile</span></code>, then the optimizer is not
allowed to modify the number or order of execution of this
<code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> with other <a class="reference internal" href="#volatile"><span class="std std-ref">volatile operations</span></a>.</p>
<p>The instruction can take an optional <code class="docutils literal notranslate"><span class="pre">align</span></code> attribute.
The alignment must be a power of two greater or equal to the size of the
<cite>&lt;value&gt;</cite> type. If unspecified, the alignment is assumed to be equal to the
size of the ‘&lt;value&gt;’ type. Note that this default alignment assumption is
different from the alignment used for the load/store instructions when align
isn’t specified.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">atomicrmw</span></code> instruction can also take an optional
“<a class="reference internal" href="#syncscope"><span class="std std-ref">syncscope</span></a>” argument.</p>
</div>
<div class="section" id="id229">
<h5><a class="toc-backref" href="#id1998">Semantics:</a><a class="headerlink" href="#id229" title="Permalink to this headline">¶</a></h5>
<p>The contents of memory at the location specified by the ‘<code class="docutils literal notranslate"><span class="pre">&lt;pointer&gt;</span></code>’
operand are atomically read, modified, and written back. The original
value at the location is returned. The modification is specified by the
operation argument:</p>
<ul class="simple">
<li>xchg: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">val</span></code></li>
<li>add: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">+</span> <span class="pre">val</span></code></li>
<li>sub: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span></code></li>
<li>and: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&amp;</span> <span class="pre">val</span></code></li>
<li>nand: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">~(*ptr</span> <span class="pre">&amp;</span> <span class="pre">val)</span></code></li>
<li>or: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">|</span> <span class="pre">val</span></code></li>
<li>xor: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">^</span> <span class="pre">val</span></code></li>
<li>max: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&gt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using a signed comparison)</li>
<li>min: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&lt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using a signed comparison)</li>
<li>umax: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&gt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using an unsigned comparison)</li>
<li>umin: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">&lt;</span> <span class="pre">val</span> <span class="pre">?</span> <span class="pre">*ptr</span> <span class="pre">:</span> <span class="pre">val</span></code> (using an unsigned comparison)</li>
<li>fadd: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">+</span> <span class="pre">val</span></code> (using floating point arithmetic)</li>
<li>fsub: <code class="docutils literal notranslate"><span class="pre">*ptr</span> <span class="pre">=</span> <span class="pre">*ptr</span> <span class="pre">-</span> <span class="pre">val</span></code> (using floating point arithmetic)</li>
</ul>
</div>
<div class="section" id="id230">
<h5><a class="toc-backref" href="#id1999">Example:</a><a class="headerlink" href="#id230" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%old</span> <span class="p">=</span> <span class="k">atomicrmw</span> <span class="k">add</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%ptr</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span> <span class="k">acquire</span>                        <span class="c">; yields i32</span>
</pre></div>
</div>
</div>
`,
                tooltip: `There are three arguments to the ‘atomicrmw’ instruction: anoperation to apply, an address whose value to modify, an argument to the
operation. The operation must be one of the following keywords:`,
            };
        case 'GETELEMENTPTR':
            return {
                url: `https://llvm.org/docs/LangRef.html#getelementptr-instruction`,
                html: `<span id="i-getelementptr"></span><h4><a class="toc-backref" href="#id2000">‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ Instruction</a><a class="headerlink" href="#getelementptr-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id231">
<h5><a class="toc-backref" href="#id2001">Syntax:</a><a class="headerlink" href="#id231" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;*</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="p">[</span><span class="n">inrange</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="n">inbounds</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;*</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">{,</span> <span class="p">[</span><span class="n">inrange</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span><span class="p">}</span><span class="o">*</span>
<span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">getelementptr</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ptr</span> <span class="n">vector</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span><span class="p">,</span> <span class="p">[</span><span class="n">inrange</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">vector</span> <span class="n">index</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">idx</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id232">
<h5><a class="toc-backref" href="#id2002">Overview:</a><a class="headerlink" href="#id232" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction is used to get the address of a
subelement of an <a class="reference internal" href="#t-aggregate"><span class="std std-ref">aggregate</span></a> data structure. It performs
address calculation only and does not access memory. The instruction can also
be used to calculate a vector of such addresses.</p>
</div>
<div class="section" id="id233">
<h5><a class="toc-backref" href="#id2003">Arguments:</a><a class="headerlink" href="#id233" title="Permalink to this headline">¶</a></h5>
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
<div class="highlight-c notranslate"><div class="highlight"><pre><span></span><span class="k">struct</span> <span class="nc">RT</span> <span class="p">{</span>
  <span class="kt">char</span> <span class="n">A</span><span class="p">;</span>
  <span class="kt">int</span> <span class="n">B</span><span class="p">[</span><span class="mi">10</span><span class="p">][</span><span class="mi">20</span><span class="p">];</span>
  <span class="kt">char</span> <span class="n">C</span><span class="p">;</span>
<span class="p">};</span>
<span class="k">struct</span> <span class="nc">ST</span> <span class="p">{</span>
  <span class="kt">int</span> <span class="n">X</span><span class="p">;</span>
  <span class="kt">double</span> <span class="n">Y</span><span class="p">;</span>
  <span class="k">struct</span> <span class="nc">RT</span> <span class="n">Z</span><span class="p">;</span>
<span class="p">};</span>

<span class="kt">int</span> <span class="o">*</span><span class="nf">foo</span><span class="p">(</span><span class="k">struct</span> <span class="nc">ST</span> <span class="o">*</span><span class="n">s</span><span class="p">)</span> <span class="p">{</span>
  <span class="k">return</span> <span class="o">&amp;</span><span class="n">s</span><span class="p">[</span><span class="mi">1</span><span class="p">].</span><span class="n">Z</span><span class="p">.</span><span class="n">B</span><span class="p">[</span><span class="mi">5</span><span class="p">][</span><span class="mi">13</span><span class="p">];</span>
<span class="p">}</span>
</pre></div>
</div>
<p>The LLVM code generated by Clang is:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%struct.RT</span> <span class="p">=</span> <span class="k">type</span> <span class="p">{</span> <span class="kt">i8</span><span class="p">,</span> <span class="p">[</span><span class="m">10</span> <span class="p">x</span> <span class="p">[</span><span class="m">20</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">]],</span> <span class="kt">i8</span> <span class="p">}</span>
<span class="nv">%struct.ST</span> <span class="p">=</span> <span class="k">type</span> <span class="p">{</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">double</span><span class="p">,</span> <span class="nv">%struct.RT</span> <span class="p">}</span>

<span class="k">define</span> <span class="kt">i32</span><span class="p">*</span> <span class="vg">@foo</span><span class="p">(</span><span class="nv">%struct.ST</span><span class="p">*</span> <span class="nv">%s</span><span class="p">)</span> <span class="k">nounwind</span> <span class="k">uwtable</span> <span class="k">readnone</span> <span class="k">optsize</span> <span class="k">ssp</span> <span class="p">{</span>
<span class="nl">entry:</span>
  <span class="nv">%arrayidx</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="k">inbounds</span> <span class="nv">%struct.ST</span><span class="p">,</span> <span class="nv">%struct.ST</span><span class="p">*</span> <span class="nv">%s</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">5</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">13</span>
  <span class="k">ret</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%arrayidx</span>
<span class="p">}</span>
</pre></div>
</div>
</div>
<div class="section" id="id234">
<h5><a class="toc-backref" href="#id2004">Semantics:</a><a class="headerlink" href="#id234" title="Permalink to this headline">¶</a></h5>
<p>In the example above, the first index is indexing into the
‘<code class="docutils literal notranslate"><span class="pre">%struct.ST*</span></code>’ type, which is a pointer, yielding a ‘<code class="docutils literal notranslate"><span class="pre">%struct.ST</span></code>’
= ‘<code class="docutils literal notranslate"><span class="pre">{</span> <span class="pre">i32,</span> <span class="pre">double,</span> <span class="pre">%struct.RT</span> <span class="pre">}</span></code>’ type, a structure. The second index
indexes into the third element of the structure, yielding a
‘<code class="docutils literal notranslate"><span class="pre">%struct.RT</span></code>’ = ‘<code class="docutils literal notranslate"><span class="pre">{</span> <span class="pre">i8</span> <span class="pre">,</span> <span class="pre">[10</span> <span class="pre">x</span> <span class="pre">[20</span> <span class="pre">x</span> <span class="pre">i32]],</span> <span class="pre">i8</span> <span class="pre">}</span></code>’ type, another
structure. The third index indexes into the second element of the
structure, yielding a ‘<code class="docutils literal notranslate"><span class="pre">[10</span> <span class="pre">x</span> <span class="pre">[20</span> <span class="pre">x</span> <span class="pre">i32]]</span></code>’ type, an array. The two
dimensions of the array are subscripted into, yielding an ‘<code class="docutils literal notranslate"><span class="pre">i32</span></code>’
type. The ‘<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>’ instruction returns a pointer to this
element, thus computing a value of ‘<code class="docutils literal notranslate"><span class="pre">i32*</span></code>’ type.</p>
<p>Note that it is perfectly legal to index partially through a structure,
returning a pointer to an inner element. Because of this, the LLVM code
for the given testcase is equivalent to:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">define</span> <span class="kt">i32</span><span class="p">*</span> <span class="vg">@foo</span><span class="p">(</span><span class="nv">%struct.ST</span><span class="p">*</span> <span class="nv">%s</span><span class="p">)</span> <span class="p">{</span>
  <span class="nv">%t1</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="nv">%struct.ST</span><span class="p">,</span> <span class="nv">%struct.ST</span><span class="p">*</span> <span class="nv">%s</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span>                        <span class="c">; yields %struct.ST*:%t1</span>
  <span class="nv">%t2</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="nv">%struct.ST</span><span class="p">,</span> <span class="nv">%struct.ST</span><span class="p">*</span> <span class="nv">%t1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">2</span>                <span class="c">; yields %struct.RT*:%t2</span>
  <span class="nv">%t3</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="nv">%struct.RT</span><span class="p">,</span> <span class="nv">%struct.RT</span><span class="p">*</span> <span class="nv">%t2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span>                <span class="c">; yields [10 x [20 x i32]]*:%t3</span>
  <span class="nv">%t4</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">[</span><span class="m">10</span> <span class="p">x</span> <span class="p">[</span><span class="m">20</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">]],</span> <span class="p">[</span><span class="m">10</span> <span class="p">x</span> <span class="p">[</span><span class="m">20</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">]]*</span> <span class="nv">%t3</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">5</span>  <span class="c">; yields [20 x i32]*:%t4</span>
  <span class="nv">%t5</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">[</span><span class="m">20</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">],</span> <span class="p">[</span><span class="m">20</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">]*</span> <span class="nv">%t4</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">13</span>               <span class="c">; yields i32*:%t5</span>
  <span class="k">ret</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%t5</span>
<span class="p">}</span>
</pre></div>
</div>
<p>If the <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> keyword is present, the result value of the
<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a> if one of the
following rules is violated:</p>
<ul class="simple">
<li>The base pointer has an <em>in bounds</em> address of an allocated object, which
means that it points into an allocated object, or to its end. The only
<em>in bounds</em> address for a null pointer in the default address-space is the
null pointer itself.</li>
<li>If the type of an index is larger than the pointer index type, the
truncation to the pointer index type preserves the signed value.</li>
<li>The multiplication of an index by the type size does not wrap the pointer
index type in a signed sense (<code class="docutils literal notranslate"><span class="pre">nsw</span></code>).</li>
<li>The successive addition of offsets (without adding the base address) does
not wrap the pointer index type in a signed sense (<code class="docutils literal notranslate"><span class="pre">nsw</span></code>).</li>
<li>The successive addition of the current address, interpreted as an unsigned
number, and an offset, interpreted as a signed number, does not wrap the
unsigned address space and remains <em>in bounds</em> of the allocated object.
As a corollary, if the added offset is non-negative, the addition does not
wrap in an unsigned sense (<code class="docutils literal notranslate"><span class="pre">nuw</span></code>).</li>
<li>In cases where the base is a vector of pointers, the <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> keyword
applies to each of the computations element-wise.</li>
</ul>
<p>These rules are based on the assumption that no allocated object may cross
the unsigned address space boundary, and no allocated object may be larger
than half the pointer index type space.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">inbounds</span></code> keyword is not present, the offsets are added to the
base address with silently-wrapping two’s complement arithmetic. If the
offsets have a different width from the pointer, they are sign-extended
or truncated to the width of the pointer. The result value of the
<code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> may be outside the object pointed to by the base
pointer. The result value may not necessarily be used to access memory
though, even if it happens to point into allocated storage. See the
<a class="reference internal" href="#pointeraliasing"><span class="std std-ref">Pointer Aliasing Rules</span></a> section for more
information.</p>
<p>If the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword is present before any index, loading from or
storing to any pointer derived from the <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> has undefined
behavior if the load or store would access memory outside of the bounds of
the element selected by the index marked as <code class="docutils literal notranslate"><span class="pre">inrange</span></code>. The result of a
pointer comparison or <code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code> (including <code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>-like operations
involving memory) involving a pointer derived from a <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> with
the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword is undefined, with the exception of comparisons
in the case where both operands are in the range of the element selected
by the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword, inclusive of the address one past the end of
that element. Note that the <code class="docutils literal notranslate"><span class="pre">inrange</span></code> keyword is currently only allowed
in constant <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> expressions.</p>
<p>The getelementptr instruction is often confusing. For some more insight
into how it works, see <a class="reference internal" href="GetElementPtr.html"><span class="doc">the getelementptr FAQ</span></a>.</p>
</div>
<div class="section" id="id235">
<h5><a class="toc-backref" href="#id2005">Example:</a><a class="headerlink" href="#id235" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; yields [12 x i8]*:aptr</span>
<span class="nv">%aptr</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="p">[</span><span class="m">12</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">]},</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="p">[</span><span class="m">12</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">]}*</span> <span class="nv">%saptr</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span>
<span class="c">; yields i8*:vptr</span>
<span class="nv">%vptr</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">&gt;},</span> <span class="p">{</span><span class="kt">i32</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">&gt;}*</span> <span class="nv">%svptr</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span>
<span class="c">; yields i8*:eptr</span>
<span class="nv">%eptr</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">[</span><span class="m">12</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">],</span> <span class="p">[</span><span class="m">12</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">]*</span> <span class="nv">%aptr</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span>
<span class="c">; yields i32*:iptr</span>
<span class="nv">%iptr</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="p">[</span><span class="m">10</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">],</span> <span class="p">[</span><span class="m">10</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">]*</span> <span class="vg">@arr</span><span class="p">,</span> <span class="kt">i16</span> <span class="m">0</span><span class="p">,</span> <span class="kt">i16</span> <span class="m">0</span>
</pre></div>
</div>
</div>
<div class="section" id="vector-of-pointers">
<h5><a class="toc-backref" href="#id2006">Vector of pointers:</a><a class="headerlink" href="#vector-of-pointers" title="Permalink to this headline">¶</a></h5>
<p>The <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code> returns a vector of pointers, instead of a single address,
when one or more of its arguments is a vector. In such cases, all vector
arguments should have the same number of elements, and every scalar argument
will be effectively broadcast into a vector during address calculation.</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; All arguments are vectors:</span>
<span class="c">;   A[i] = ptrs[i] + offsets[i]*sizeof(i8)</span>
<span class="nv">%A</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="kt">i8</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">*&gt;</span> <span class="nv">%ptrs</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span> <span class="nv">%offsets</span>

<span class="c">; Add the same scalar offset to each pointer of a vector:</span>
<span class="c">;   A[i] = ptrs[i] + offset*sizeof(i8)</span>
<span class="nv">%A</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="kt">i8</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">*&gt;</span> <span class="nv">%ptrs</span><span class="p">,</span> <span class="kt">i64</span> <span class="nv">%offset</span>

<span class="c">; Add distinct offsets to the same pointer:</span>
<span class="c">;   A[i] = ptr + offsets[i]*sizeof(i8)</span>
<span class="nv">%A</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="kt">i8</span><span class="p">,</span> <span class="kt">i8</span><span class="p">*</span> <span class="nv">%ptr</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span> <span class="nv">%offsets</span>

<span class="c">; In all cases described above the type of the result is &lt;4 x i8*&gt;</span>
</pre></div>
</div>
<p>The two following instructions are equivalent:</p>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="k">getelementptr</span>  <span class="nv">%struct.ST</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="nv">%struct.ST</span><span class="p">*&gt;</span> <span class="nv">%s</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span> <span class="nv">%ind1</span><span class="p">,</span>
  <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">2</span><span class="p">&gt;,</span>
  <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">&gt;,</span>
  <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="nv">%ind4</span><span class="p">,</span>
  <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i64</span> <span class="m">13</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">13</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">13</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">13</span><span class="p">&gt;</span>

<span class="k">getelementptr</span>  <span class="nv">%struct.ST</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="nv">%struct.ST</span><span class="p">*&gt;</span> <span class="nv">%s</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span> <span class="nv">%ind1</span><span class="p">,</span>
  <span class="kt">i32</span> <span class="m">2</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">1</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="nv">%ind4</span><span class="p">,</span> <span class="kt">i64</span> <span class="m">13</span>
</pre></div>
</div>
<p>Let’s look at the C code, where the vector version of <code class="docutils literal notranslate"><span class="pre">getelementptr</span></code>
makes sense:</p>
<div class="highlight-c notranslate"><div class="highlight"><pre><span></span><span class="c1">// Let's assume that we vectorize the following loop:</span>
<span class="kt">double</span> <span class="o">*</span><span class="n">A</span><span class="p">,</span> <span class="o">*</span><span class="n">B</span><span class="p">;</span> <span class="kt">int</span> <span class="o">*</span><span class="n">C</span><span class="p">;</span>
<span class="k">for</span> <span class="p">(</span><span class="kt">int</span> <span class="n">i</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span> <span class="n">i</span> <span class="o">&lt;</span> <span class="n">size</span><span class="p">;</span> <span class="o">++</span><span class="n">i</span><span class="p">)</span> <span class="p">{</span>
  <span class="n">A</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">B</span><span class="p">[</span><span class="n">C</span><span class="p">[</span><span class="n">i</span><span class="p">]];</span>
<span class="p">}</span>
</pre></div>
</div>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">; get pointers for 8 elements from array B</span>
<span class="nv">%ptrs</span> <span class="p">=</span> <span class="k">getelementptr</span> <span class="kt">double</span><span class="p">,</span> <span class="kt">double</span><span class="p">*</span> <span class="nv">%B</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">8</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="nv">%C</span>
<span class="c">; load 8 elements from array B into A</span>
<span class="nv">%A</span> <span class="p">=</span> <span class="k">call</span> <span class="p">&lt;</span><span class="m">8</span> <span class="p">x</span> <span class="kt">double</span><span class="p">&gt;</span> <span class="vg">@llvm.masked.gather.v8f64.v8p0f64</span><span class="p">(&lt;</span><span class="m">8</span> <span class="p">x</span> <span class="kt">double</span><span class="p">*&gt;</span> <span class="nv">%ptrs</span><span class="p">,</span>
     <span class="kt">i32</span> <span class="m">8</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">8</span> <span class="p">x</span> <span class="kt">i1</span><span class="p">&gt;</span> <span class="nv">%mask</span><span class="p">,</span> <span class="p">&lt;</span><span class="m">8</span> <span class="p">x</span> <span class="kt">double</span><span class="p">&gt;</span> <span class="nv">%passthru</span><span class="p">)</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The first argument is always a type used as the basis for the calculations.The second argument is always a pointer or a vector of pointers, and is the
base address to start from. The remaining arguments are indices
that indicate which of the elements of the aggregate object are indexed.
The interpretation of each index is dependent on the type being indexed
into. The first index always indexes the pointer value given as the
second argument, the second index indexes a value of the type pointed to
(not necessarily the value directly pointed to, since the first index
can be non-zero), etc. The first type indexed into must be a pointer
value, subsequent types can be arrays, vectors, and structs. Note that
subsequent types being indexed into can never be pointers, since that
would require loading the pointer before continuing calculation.`,
            };
        case 'TRUNC-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#trunc-to-instruction`,
                html: `<span id="i-trunc"></span><h4><a class="toc-backref" href="#id2008">‘<code class="docutils literal notranslate"><span class="pre">trunc</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#trunc-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id236">
<h5><a class="toc-backref" href="#id2009">Syntax:</a><a class="headerlink" href="#id236" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">trunc</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id237">
<h5><a class="toc-backref" href="#id2010">Overview:</a><a class="headerlink" href="#id237" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction truncates its operand to the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id238">
<h5><a class="toc-backref" href="#id2011">Arguments:</a><a class="headerlink" href="#id238" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction takes a value to trunc, and a type to trunc
it to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors
of the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
larger than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Equal sized
types are not allowed.</p>
</div>
<div class="section" id="id239">
<h5><a class="toc-backref" href="#id2012">Semantics:</a><a class="headerlink" href="#id239" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">trunc</span></code>’ instruction truncates the high order bits in <code class="docutils literal notranslate"><span class="pre">value</span></code>
and converts the remaining bits to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Since the source size must
be larger than the destination size, <code class="docutils literal notranslate"><span class="pre">trunc</span></code> cannot be a <em>no-op cast</em>.
It will always truncate bits.</p>
</div>
<div class="section" id="id240">
<h5><a class="toc-backref" href="#id2013">Example:</a><a class="headerlink" href="#id240" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">trunc</span> <span class="kt">i32</span> <span class="m">257</span> <span class="k">to</span> <span class="kt">i8</span>                        <span class="c">; yields i8:1</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">trunc</span> <span class="kt">i32</span> <span class="m">123</span> <span class="k">to</span> <span class="kt">i1</span>                        <span class="c">; yields i1:true</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">trunc</span> <span class="kt">i32</span> <span class="m">122</span> <span class="k">to</span> <span class="kt">i1</span>                        <span class="c">; yields i1:false</span>
<span class="nv">%W</span> <span class="p">=</span> <span class="k">trunc</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i16</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i16</span> <span class="m">8</span><span class="p">,</span> <span class="kt">i16</span> <span class="m">7</span><span class="p">&gt;</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">&gt;</span> <span class="c">; yields &lt;i8 8, i8 7&gt;</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘trunc’ instruction takes a value to trunc, and a type to truncit to. Both types must be of integer types, or vectors
of the same number of integers. The bit size of the value must be
larger than the bit size of the destination type, ty2. Equal sized
types are not allowed.`,
            };
        case 'ZEXT-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#zext-to-instruction`,
                html: `<span id="i-zext"></span><h4><a class="toc-backref" href="#id2014">‘<code class="docutils literal notranslate"><span class="pre">zext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#zext-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id241">
<h5><a class="toc-backref" href="#id2015">Syntax:</a><a class="headerlink" href="#id241" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">zext</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id242">
<h5><a class="toc-backref" href="#id2016">Overview:</a><a class="headerlink" href="#id242" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">zext</span></code>’ instruction zero extends its operand to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id243">
<h5><a class="toc-backref" href="#id2017">Arguments:</a><a class="headerlink" href="#id243" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">zext</span></code>’ instruction takes a value to cast, and a type to cast it
to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors of
the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
smaller than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id244">
<h5><a class="toc-backref" href="#id2018">Semantics:</a><a class="headerlink" href="#id244" title="Permalink to this headline">¶</a></h5>
<p>The <code class="docutils literal notranslate"><span class="pre">zext</span></code> fills the high order bits of the <code class="docutils literal notranslate"><span class="pre">value</span></code> with zero bits
until it reaches the size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
<p>When zero extending from i1, the result will always be either 0 or 1.</p>
</div>
<div class="section" id="id245">
<h5><a class="toc-backref" href="#id2019">Example:</a><a class="headerlink" href="#id245" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">zext</span> <span class="kt">i32</span> <span class="m">257</span> <span class="k">to</span> <span class="kt">i64</span>              <span class="c">; yields i64:257</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">zext</span> <span class="kt">i1</span> <span class="k">true</span> <span class="k">to</span> <span class="kt">i32</span>              <span class="c">; yields i32:1</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">zext</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i16</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i16</span> <span class="m">8</span><span class="p">,</span> <span class="kt">i16</span> <span class="m">7</span><span class="p">&gt;</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="c">; yields &lt;i32 8, i32 7&gt;</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘zext’ instruction takes a value to cast, and a type to cast itto. Both types must be of integer types, or vectors of
the same number of integers. The bit size of the value must be
smaller than the bit size of the destination type, ty2.`,
            };
        case 'SEXT-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#sext-to-instruction`,
                html: `<span id="i-sext"></span><h4><a class="toc-backref" href="#id2020">‘<code class="docutils literal notranslate"><span class="pre">sext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#sext-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id246">
<h5><a class="toc-backref" href="#id2021">Syntax:</a><a class="headerlink" href="#id246" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sext</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id247">
<h5><a class="toc-backref" href="#id2022">Overview:</a><a class="headerlink" href="#id247" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ sign extends <code class="docutils literal notranslate"><span class="pre">value</span></code> to the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id248">
<h5><a class="toc-backref" href="#id2023">Arguments:</a><a class="headerlink" href="#id248" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ instruction takes a value to cast, and a type to cast it
to. Both types must be of <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> types, or vectors of
the same number of integers. The bit size of the <code class="docutils literal notranslate"><span class="pre">value</span></code> must be
smaller than the bit size of the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id249">
<h5><a class="toc-backref" href="#id2024">Semantics:</a><a class="headerlink" href="#id249" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sext</span></code>’ instruction performs a sign extension by copying the sign
bit (highest order bit) of the <code class="docutils literal notranslate"><span class="pre">value</span></code> until it reaches the bit size
of the type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
<p>When sign extending from i1, the extension always results in -1 or 0.</p>
</div>
<div class="section" id="id250">
<h5><a class="toc-backref" href="#id2025">Example:</a><a class="headerlink" href="#id250" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">sext</span> <span class="kt">i8</span>  <span class="m">-1</span> <span class="k">to</span> <span class="kt">i16</span>              <span class="c">; yields i16   :65535</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">sext</span> <span class="kt">i1</span> <span class="k">true</span> <span class="k">to</span> <span class="kt">i32</span>             <span class="c">; yields i32:-1</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">sext</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i16</span><span class="p">&gt;</span> <span class="p">&lt;</span><span class="kt">i16</span> <span class="m">8</span><span class="p">,</span> <span class="kt">i16</span> <span class="m">7</span><span class="p">&gt;</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">2</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="c">; yields &lt;i32 8, i32 7&gt;</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘sext’ instruction takes a value to cast, and a type to cast itto. Both types must be of integer types, or vectors of
the same number of integers. The bit size of the value must be
smaller than the bit size of the destination type, ty2.`,
            };
        case 'FPTRUNC-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptrunc-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2026">‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#fptrunc-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id251">
<h5><a class="toc-backref" href="#id2027">Syntax:</a><a class="headerlink" href="#id251" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptrunc</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id252">
<h5><a class="toc-backref" href="#id2028">Overview:</a><a class="headerlink" href="#id252" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction truncates <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id253">
<h5><a class="toc-backref" href="#id2029">Arguments:</a><a class="headerlink" href="#id253" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction takes a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
value to cast and a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to cast it to.
The size of <code class="docutils literal notranslate"><span class="pre">value</span></code> must be larger than the size of <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. This
implies that <code class="docutils literal notranslate"><span class="pre">fptrunc</span></code> cannot be used to make a <em>no-op cast</em>.</p>
</div>
<div class="section" id="id254">
<h5><a class="toc-backref" href="#id2030">Semantics:</a><a class="headerlink" href="#id254" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptrunc</span></code>’ instruction casts a <code class="docutils literal notranslate"><span class="pre">value</span></code> from a larger
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to a smaller <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type.
This instruction is assumed to execute in the default <a class="reference internal" href="#floatenv"><span class="std std-ref">floating-point
environment</span></a>.</p>
</div>
<div class="section" id="id255">
<h5><a class="toc-backref" href="#id2031">Example:</a><a class="headerlink" href="#id255" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">fptrunc</span> <span class="kt">double</span> <span class="m">16777217.0</span> <span class="k">to</span> <span class="kt">float</span>    <span class="c">; yields float:16777216.0</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">fptrunc</span> <span class="kt">double</span> <span class="m">1.0E+300</span> <span class="k">to</span> <span class="kt">half</span>       <span class="c">; yields half:+infinity</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘fptrunc’ instruction takes a floating-pointvalue to cast and a floating-point type to cast it to.
The size of value must be larger than the size of ty2. This
implies that fptrunc cannot be used to make a no-op cast.`,
            };
        case 'FPEXT-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#fpext-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2032">‘<code class="docutils literal notranslate"><span class="pre">fpext</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#fpext-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id256">
<h5><a class="toc-backref" href="#id2033">Syntax:</a><a class="headerlink" href="#id256" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fpext</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id257">
<h5><a class="toc-backref" href="#id2034">Overview:</a><a class="headerlink" href="#id257" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ extends a floating-point <code class="docutils literal notranslate"><span class="pre">value</span></code> to a larger floating-point
value.</p>
</div>
<div class="section" id="id258">
<h5><a class="toc-backref" href="#id2035">Arguments:</a><a class="headerlink" href="#id258" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ instruction takes a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
<code class="docutils literal notranslate"><span class="pre">value</span></code> to cast, and a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to cast it
to. The source type must be smaller than the destination type.</p>
</div>
<div class="section" id="id259">
<h5><a class="toc-backref" href="#id2036">Semantics:</a><a class="headerlink" href="#id259" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fpext</span></code>’ instruction extends the <code class="docutils literal notranslate"><span class="pre">value</span></code> from a smaller
<a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type to a larger <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. The <code class="docutils literal notranslate"><span class="pre">fpext</span></code> cannot be used to make a
<em>no-op cast</em> because it always changes bits. Use <code class="docutils literal notranslate"><span class="pre">bitcast</span></code> to make a
<em>no-op cast</em> for a floating-point cast.</p>
</div>
<div class="section" id="id260">
<h5><a class="toc-backref" href="#id2037">Example:</a><a class="headerlink" href="#id260" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">fpext</span> <span class="kt">float</span> <span class="m">3.125</span> <span class="k">to</span> <span class="kt">double</span>         <span class="c">; yields double:3.125000e+00</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">fpext</span> <span class="kt">double</span> <span class="nv">%X</span> <span class="k">to</span> <span class="kt">fp128</span>            <span class="c">; yields fp128:0xL00000000000000004000900000000000</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘fpext’ instruction takes a floating-pointvalue to cast, and a floating-point type to cast it
to. The source type must be smaller than the destination type.`,
            };
        case 'FPTOUI-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptoui-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2038">‘<code class="docutils literal notranslate"><span class="pre">fptoui</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#fptoui-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id261">
<h5><a class="toc-backref" href="#id2039">Syntax:</a><a class="headerlink" href="#id261" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptoui</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id262">
<h5><a class="toc-backref" href="#id2040">Overview:</a><a class="headerlink" href="#id262" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ converts a floating-point <code class="docutils literal notranslate"><span class="pre">value</span></code> to its unsigned
integer equivalent of type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id263">
<h5><a class="toc-backref" href="#id2041">Arguments:</a><a class="headerlink" href="#id263" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> value, and a type to
cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector floating-point type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector integer
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</div>
<div class="section" id="id264">
<h5><a class="toc-backref" href="#id2042">Semantics:</a><a class="headerlink" href="#id264" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptoui</span></code>’ instruction converts its <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> operand into the nearest (rounding towards zero)
unsigned integer value. If the value cannot fit in <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</div>
<div class="section" id="id265">
<h5><a class="toc-backref" href="#id2043">Example:</a><a class="headerlink" href="#id265" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">fptoui</span> <span class="kt">double</span> <span class="m">123.0</span> <span class="k">to</span> <span class="kt">i32</span>      <span class="c">; yields i32:123</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">fptoui</span> <span class="kt">float</span> <span class="m">1.0E+300</span> <span class="k">to</span> <span class="kt">i1</span>     <span class="c">; yields undefined:1</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">fptoui</span> <span class="kt">float</span> <span class="m">1.04E+17</span> <span class="k">to</span> <span class="kt">i8</span>     <span class="c">; yields undefined:1</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘fptoui’ instruction takes a value to cast, which must be ascalar or vector floating-point value, and a type to
cast it to ty2, which must be an integer type. If
ty is a vector floating-point type, ty2 must be a vector integer
type with the same number of elements as ty`,
            };
        case 'FPTOSI-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#fptosi-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2044">‘<code class="docutils literal notranslate"><span class="pre">fptosi</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#fptosi-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id266">
<h5><a class="toc-backref" href="#id2045">Syntax:</a><a class="headerlink" href="#id266" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fptosi</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id267">
<h5><a class="toc-backref" href="#id2046">Overview:</a><a class="headerlink" href="#id267" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction converts <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a>
<code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id268">
<h5><a class="toc-backref" href="#id2047">Arguments:</a><a class="headerlink" href="#id268" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> value, and a type to
cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector floating-point type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector integer
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</div>
<div class="section" id="id269">
<h5><a class="toc-backref" href="#id2048">Semantics:</a><a class="headerlink" href="#id269" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fptosi</span></code>’ instruction converts its <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> operand into the nearest (rounding towards zero)
signed integer value. If the value cannot fit in <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, the result
is a <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison value</span></a>.</p>
</div>
<div class="section" id="id270">
<h5><a class="toc-backref" href="#id2049">Example:</a><a class="headerlink" href="#id270" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">fptosi</span> <span class="kt">double</span> <span class="m">-123.0</span> <span class="k">to</span> <span class="kt">i32</span>      <span class="c">; yields i32:-123</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">fptosi</span> <span class="kt">float</span> <span class="m">1.0E-247</span> <span class="k">to</span> <span class="kt">i1</span>      <span class="c">; yields undefined:1</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">fptosi</span> <span class="kt">float</span> <span class="m">1.04E+17</span> <span class="k">to</span> <span class="kt">i8</span>      <span class="c">; yields undefined:1</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘fptosi’ instruction takes a value to cast, which must be ascalar or vector floating-point value, and a type to
cast it to ty2, which must be an integer type. If
ty is a vector floating-point type, ty2 must be a vector integer
type with the same number of elements as ty`,
            };
        case 'UITOFP-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#uitofp-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2050">‘<code class="docutils literal notranslate"><span class="pre">uitofp</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#uitofp-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id271">
<h5><a class="toc-backref" href="#id2051">Syntax:</a><a class="headerlink" href="#id271" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">uitofp</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id272">
<h5><a class="toc-backref" href="#id2052">Overview:</a><a class="headerlink" href="#id272" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction regards <code class="docutils literal notranslate"><span class="pre">value</span></code> as an unsigned integer
and converts that value to the <code class="docutils literal notranslate"><span class="pre">ty2</span></code> type.</p>
</div>
<div class="section" id="id273">
<h5><a class="toc-backref" href="#id2053">Arguments:</a><a class="headerlink" href="#id273" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> value, and a type to cast it to
<code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector integer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector floating-point
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</div>
<div class="section" id="id274">
<h5><a class="toc-backref" href="#id2054">Semantics:</a><a class="headerlink" href="#id274" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">uitofp</span></code>’ instruction interprets its operand as an unsigned
integer quantity and converts it to the corresponding floating-point
value. If the value cannot be exactly represented, it is rounded using
the default rounding mode.</p>
</div>
<div class="section" id="id275">
<h5><a class="toc-backref" href="#id2055">Example:</a><a class="headerlink" href="#id275" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">uitofp</span> <span class="kt">i32</span> <span class="m">257</span> <span class="k">to</span> <span class="kt">float</span>         <span class="c">; yields float:257.0</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">uitofp</span> <span class="kt">i8</span> <span class="m">-1</span> <span class="k">to</span> <span class="kt">double</span>          <span class="c">; yields double:255.0</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘uitofp’ instruction takes a value to cast, which must be ascalar or vector integer value, and a type to cast it to
ty2, which must be an floating-point type. If
ty is a vector integer type, ty2 must be a vector floating-point
type with the same number of elements as ty`,
            };
        case 'SITOFP-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#sitofp-to-instruction`,
                html: `<h4><a class="toc-backref" href="#id2056">‘<code class="docutils literal notranslate"><span class="pre">sitofp</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#sitofp-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id276">
<h5><a class="toc-backref" href="#id2057">Syntax:</a><a class="headerlink" href="#id276" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">sitofp</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id277">
<h5><a class="toc-backref" href="#id2058">Overview:</a><a class="headerlink" href="#id277" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction regards <code class="docutils literal notranslate"><span class="pre">value</span></code> as a signed integer and
converts that value to the <code class="docutils literal notranslate"><span class="pre">ty2</span></code> type.</p>
</div>
<div class="section" id="id278">
<h5><a class="toc-backref" href="#id2059">Arguments:</a><a class="headerlink" href="#id278" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction takes a value to cast, which must be a
scalar or vector <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> value, and a type to cast it to
<code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type. If
<code class="docutils literal notranslate"><span class="pre">ty</span></code> is a vector integer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code> must be a vector floating-point
type with the same number of elements as <code class="docutils literal notranslate"><span class="pre">ty</span></code></p>
</div>
<div class="section" id="id279">
<h5><a class="toc-backref" href="#id2060">Semantics:</a><a class="headerlink" href="#id279" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">sitofp</span></code>’ instruction interprets its operand as a signed integer
quantity and converts it to the corresponding floating-point value. If the
value cannot be exactly represented, it is rounded using the default rounding
mode.</p>
</div>
<div class="section" id="id280">
<h5><a class="toc-backref" href="#id2061">Example:</a><a class="headerlink" href="#id280" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">sitofp</span> <span class="kt">i32</span> <span class="m">257</span> <span class="k">to</span> <span class="kt">float</span>         <span class="c">; yields float:257.0</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">sitofp</span> <span class="kt">i8</span> <span class="m">-1</span> <span class="k">to</span> <span class="kt">double</span>          <span class="c">; yields double:-1.0</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘sitofp’ instruction takes a value to cast, which must be ascalar or vector integer value, and a type to cast it to
ty2, which must be an floating-point type. If
ty is a vector integer type, ty2 must be a vector floating-point
type with the same number of elements as ty`,
            };
        case 'PTRTOINT-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#ptrtoint-to-instruction`,
                html: `<span id="i-ptrtoint"></span><h4><a class="toc-backref" href="#id2062">‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#ptrtoint-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id281">
<h5><a class="toc-backref" href="#id2063">Syntax:</a><a class="headerlink" href="#id281" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">ptrtoint</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id282">
<h5><a class="toc-backref" href="#id2064">Overview:</a><a class="headerlink" href="#id282" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction converts the pointer or a vector of
pointers <code class="docutils literal notranslate"><span class="pre">value</span></code> to the integer (or vector of integers) type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id283">
<h5><a class="toc-backref" href="#id2065">Arguments:</a><a class="headerlink" href="#id283" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction takes a <code class="docutils literal notranslate"><span class="pre">value</span></code> to cast, which must be
a value of type <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> or a vector of pointers, and a
type to cast it to <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, which must be an <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or
a vector of integers type.</p>
</div>
<div class="section" id="id284">
<h5><a class="toc-backref" href="#id2066">Semantics:</a><a class="headerlink" href="#id284" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">ptrtoint</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to integer type
<code class="docutils literal notranslate"><span class="pre">ty2</span></code> by interpreting the pointer value as an integer and either
truncating or zero extending that value to the size of the integer type.
If <code class="docutils literal notranslate"><span class="pre">value</span></code> is smaller than <code class="docutils literal notranslate"><span class="pre">ty2</span></code> then a zero extension is done. If
<code class="docutils literal notranslate"><span class="pre">value</span></code> is larger than <code class="docutils literal notranslate"><span class="pre">ty2</span></code> then a truncation is done. If they are
the same size, then nothing is done (<em>no-op cast</em>) other than a type
change.</p>
</div>
<div class="section" id="id285">
<h5><a class="toc-backref" href="#id2067">Example:</a><a class="headerlink" href="#id285" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">ptrtoint</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%P</span> <span class="k">to</span> <span class="kt">i8</span>                         <span class="c">; yields truncation on 32-bit architecture</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">ptrtoint</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%P</span> <span class="k">to</span> <span class="kt">i64</span>                        <span class="c">; yields zero extension on 32-bit architecture</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">ptrtoint</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">*&gt;</span> <span class="nv">%P</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i64</span><span class="p">&gt;</span><span class="c">; yields vector zero extension for a vector of addresses on 32-bit architecture</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘ptrtoint’ instruction takes a value to cast, which must bea value of type pointer or a vector of pointers, and a
type to cast it to ty2, which must be an integer or
a vector of integers type.`,
            };
        case 'INTTOPTR-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#inttoptr-to-instruction`,
                html: `<span id="i-inttoptr"></span><h4><a class="toc-backref" href="#id2068">‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#inttoptr-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id286">
<h5><a class="toc-backref" href="#id2069">Syntax:</a><a class="headerlink" href="#id286" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = inttoptr &lt;ty&gt; &lt;value&gt; to &lt;ty2&gt;[, !dereferenceable !&lt;deref_bytes_node&gt;][, !dereferenceable_or_null !&lt;deref_bytes_node&gt;]             ; yields ty2
</pre></div>
</div>
</div>
<div class="section" id="id287">
<h5><a class="toc-backref" href="#id2070">Overview:</a><a class="headerlink" href="#id287" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span></code>’ instruction converts an integer <code class="docutils literal notranslate"><span class="pre">value</span></code> to a
pointer type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>.</p>
</div>
<div class="section" id="id288">
<h5><a class="toc-backref" href="#id2071">Arguments:</a><a class="headerlink" href="#id288" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id289">
<h5><a class="toc-backref" href="#id2072">Semantics:</a><a class="headerlink" href="#id289" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">inttoptr</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code> by
applying either a zero extension or a truncation depending on the size
of the integer <code class="docutils literal notranslate"><span class="pre">value</span></code>. If <code class="docutils literal notranslate"><span class="pre">value</span></code> is larger than the size of a
pointer then a truncation is done. If <code class="docutils literal notranslate"><span class="pre">value</span></code> is smaller than the size
of a pointer then a zero extension is done. If they are the same size,
nothing is done (<em>no-op cast</em>).</p>
</div>
<div class="section" id="id290">
<h5><a class="toc-backref" href="#id2073">Example:</a><a class="headerlink" href="#id290" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">inttoptr</span> <span class="kt">i32</span> <span class="m">255</span> <span class="k">to</span> <span class="kt">i32</span><span class="p">*</span>          <span class="c">; yields zero extension on 64-bit architecture</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">inttoptr</span> <span class="kt">i32</span> <span class="m">255</span> <span class="k">to</span> <span class="kt">i32</span><span class="p">*</span>          <span class="c">; yields no-op on 32-bit architecture</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">inttoptr</span> <span class="kt">i64</span> <span class="m">0</span> <span class="k">to</span> <span class="kt">i32</span><span class="p">*</span>            <span class="c">; yields truncation on 32-bit architecture</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">inttoptr</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">&gt;</span> <span class="nv">%G</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">*&gt;</span><span class="c">; yields truncation of vector G to four pointers</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘inttoptr’ instruction takes an integer value tocast, and a type to cast it to, which must be a pointer
type.`,
            };
        case 'BITCAST-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#bitcast-to-instruction`,
                html: `<span id="i-bitcast"></span><h4><a class="toc-backref" href="#id2074">‘<code class="docutils literal notranslate"><span class="pre">bitcast</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#bitcast-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id291">
<h5><a class="toc-backref" href="#id2075">Syntax:</a><a class="headerlink" href="#id291" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">bitcast</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">ty2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id292">
<h5><a class="toc-backref" href="#id2076">Overview:</a><a class="headerlink" href="#id292" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code> without
changing any bits.</p>
</div>
<div class="section" id="id293">
<h5><a class="toc-backref" href="#id2077">Arguments:</a><a class="headerlink" href="#id293" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction takes a value to cast, which must be a
non-aggregate first class value, and a type to cast it to, which must
also be a non-aggregate <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. The
bit sizes of <code class="docutils literal notranslate"><span class="pre">value</span></code> and the destination type, <code class="docutils literal notranslate"><span class="pre">ty2</span></code>, must be
identical. If the source type is a pointer, the destination type must
also be a pointer of the same size. This instruction supports bitwise
conversion of vectors to integers and to vectors of other types (as
long as they have the same size).</p>
</div>
<div class="section" id="id294">
<h5><a class="toc-backref" href="#id2078">Semantics:</a><a class="headerlink" href="#id294" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">bitcast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">value</span></code> to type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. It
is always a <em>no-op cast</em> because no bits change with this
conversion. The conversion is done as if the <code class="docutils literal notranslate"><span class="pre">value</span></code> had been stored
to memory and read back as type <code class="docutils literal notranslate"><span class="pre">ty2</span></code>. Pointer (or vector of
pointers) types may only be converted to other pointer (or vector of
pointers) types with the same address space through this instruction.
To convert pointers to other types, use the <a class="reference internal" href="#i-inttoptr"><span class="std std-ref">inttoptr</span></a>
or <a class="reference internal" href="#i-ptrtoint"><span class="std std-ref">ptrtoint</span></a> instructions first.</p>
<p>There is a caveat for bitcasts involving vector types in relation to
endianess. For example <code class="docutils literal notranslate"><span class="pre">bitcast</span> <span class="pre">&lt;2</span> <span class="pre">x</span> <span class="pre">i8&gt;</span> <span class="pre">&lt;value&gt;</span> <span class="pre">to</span> <span class="pre">i16</span></code> puts element zero
of the vector in the least significant bits of the i16 for little-endian while
element zero ends up in the most significant bits for big-endian.</p>
</div>
<div class="section" id="id295">
<h5><a class="toc-backref" href="#id2079">Example:</a><a class="headerlink" href="#id295" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>%X = bitcast i8 255 to i8          ; yields i8 :-1
%Y = bitcast i32* %x to i16*      ; yields i16*:%x
%Z = bitcast &lt;2 x i32&gt; %V to i64;  ; yields i64: %V (depends on endianess)
%Z = bitcast &lt;2 x i32*&gt; %V to &lt;2 x i64*&gt; ; yields &lt;2 x i64*&gt;
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘bitcast’ instruction takes a value to cast, which must be anon-aggregate first class value, and a type to cast it to, which must
also be a non-aggregate first class type. The
bit sizes of value and the destination type, ty2, must be
identical. If the source type is a pointer, the destination type must
also be a pointer of the same size. This instruction supports bitwise
conversion of vectors to integers and to vectors of other types (as
long as they have the same size).`,
            };
        case 'ADDRSPACECAST-TO':
            return {
                url: `https://llvm.org/docs/LangRef.html#addrspacecast-to-instruction`,
                html: `<span id="i-addrspacecast"></span><h4><a class="toc-backref" href="#id2080">‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span> <span class="pre">..</span> <span class="pre">to</span></code>’ Instruction</a><a class="headerlink" href="#addrspacecast-to-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id296">
<h5><a class="toc-backref" href="#id2081">Syntax:</a><a class="headerlink" href="#id296" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">addrspacecast</span> <span class="o">&lt;</span><span class="n">pty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ptrval</span><span class="o">&gt;</span> <span class="n">to</span> <span class="o">&lt;</span><span class="n">pty2</span><span class="o">&gt;</span>       <span class="p">;</span> <span class="n">yields</span> <span class="n">pty2</span>
</pre></div>
</div>
</div>
<div class="section" id="id297">
<h5><a class="toc-backref" href="#id2082">Overview:</a><a class="headerlink" href="#id297" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction converts <code class="docutils literal notranslate"><span class="pre">ptrval</span></code> from <code class="docutils literal notranslate"><span class="pre">pty</span></code> in
address space <code class="docutils literal notranslate"><span class="pre">n</span></code> to type <code class="docutils literal notranslate"><span class="pre">pty2</span></code> in address space <code class="docutils literal notranslate"><span class="pre">m</span></code>.</p>
</div>
<div class="section" id="id298">
<h5><a class="toc-backref" href="#id2083">Arguments:</a><a class="headerlink" href="#id298" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction takes a pointer or vector of pointer value
to cast and a pointer type to cast it to, which must have a different
address space.</p>
</div>
<div class="section" id="id299">
<h5><a class="toc-backref" href="#id2084">Semantics:</a><a class="headerlink" href="#id299" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">addrspacecast</span></code>’ instruction converts the pointer value
<code class="docutils literal notranslate"><span class="pre">ptrval</span></code> to type <code class="docutils literal notranslate"><span class="pre">pty2</span></code>. It can be a <em>no-op cast</em> or a complex
value modification, depending on the target and the address space
pair. Pointer conversions within the same address space must be
performed with the <code class="docutils literal notranslate"><span class="pre">bitcast</span></code> instruction. Note that if the address space
conversion is legal then both result and operand refer to the same memory
location.</p>
</div>
<div class="section" id="id300">
<h5><a class="toc-backref" href="#id2085">Example:</a><a class="headerlink" href="#id300" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">addrspacecast</span> <span class="kt">i32</span><span class="p">*</span> <span class="nv">%x</span> <span class="k">to</span> <span class="kt">i32</span> <span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)*</span>    <span class="c">; yields i32 addrspace(1)*:%x</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">addrspacecast</span> <span class="kt">i32</span> <span class="k">addrspace</span><span class="p">(</span><span class="m">1</span><span class="p">)*</span> <span class="nv">%y</span> <span class="k">to</span> <span class="kt">i64</span> <span class="k">addrspace</span><span class="p">(</span><span class="m">2</span><span class="p">)*</span>    <span class="c">; yields i64 addrspace(2)*:%y</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">addrspacecast</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">i32</span><span class="p">*&gt;</span> <span class="nv">%z</span> <span class="k">to</span> <span class="p">&lt;</span><span class="m">4</span> <span class="p">x</span> <span class="kt">float</span> <span class="k">addrspace</span><span class="p">(</span><span class="m">3</span><span class="p">)*&gt;</span>   <span class="c">; yields &lt;4 x float addrspace(3)*&gt;:%z</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘addrspacecast’ instruction takes a pointer or vector of pointer valueto cast and a pointer type to cast it to, which must have a different
address space.`,
            };
        case 'ICMP':
            return {
                url: `https://llvm.org/docs/LangRef.html#icmp-instruction`,
                html: `<span id="i-icmp"></span><h4><a class="toc-backref" href="#id2087">‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ Instruction</a><a class="headerlink" href="#icmp-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id301">
<h5><a class="toc-backref" href="#id2088">Syntax:</a><a class="headerlink" href="#id301" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">icmp</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>   <span class="p">;</span> <span class="n">yields</span> <span class="n">i1</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id302">
<h5><a class="toc-backref" href="#id2089">Overview:</a><a class="headerlink" href="#id302" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ instruction returns a boolean value or a vector of
boolean values based on comparison of its two integer, integer vector,
pointer, or pointer vector operands.</p>
</div>
<div class="section" id="id303">
<h5><a class="toc-backref" href="#id2090">Arguments:</a><a class="headerlink" href="#id303" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ instruction takes three operands. The first operand is
the condition code indicating the kind of comparison to perform. It is
not a value, just a keyword. The possible condition codes are:</p>
<ol class="arabic simple" id="icmp-md-cc">
<li><code class="docutils literal notranslate"><span class="pre">eq</span></code>: equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ne</span></code>: not equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: unsigned greater than</li>
<li><code class="docutils literal notranslate"><span class="pre">uge</span></code>: unsigned greater or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ult</span></code>: unsigned less than</li>
<li><code class="docutils literal notranslate"><span class="pre">ule</span></code>: unsigned less or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">sgt</span></code>: signed greater than</li>
<li><code class="docutils literal notranslate"><span class="pre">sge</span></code>: signed greater or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">slt</span></code>: signed less than</li>
<li><code class="docutils literal notranslate"><span class="pre">sle</span></code>: signed less or equal</li>
</ol>
<p>The remaining two arguments must be <a class="reference internal" href="#t-integer"><span class="std std-ref">integer</span></a> or
<a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> or integer <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> typed. They
must also be identical types.</p>
</div>
<div class="section" id="id304">
<h5><a class="toc-backref" href="#id2091">Semantics:</a><a class="headerlink" href="#id304" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">icmp</span></code>’ compares <code class="docutils literal notranslate"><span class="pre">op1</span></code> and <code class="docutils literal notranslate"><span class="pre">op2</span></code> according to the condition
code given as <code class="docutils literal notranslate"><span class="pre">cond</span></code>. The comparison performed always yields either an
<a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a> or vector of <code class="docutils literal notranslate"><span class="pre">i1</span></code> result, as follows:</p>
<ol class="arabic simple" id="icmp-md-cc-sem">
<li><code class="docutils literal notranslate"><span class="pre">eq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if the operands are equal, <code class="docutils literal notranslate"><span class="pre">false</span></code>
otherwise. No sign interpretation is necessary or performed.</li>
<li><code class="docutils literal notranslate"><span class="pre">ne</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if the operands are unequal, <code class="docutils literal notranslate"><span class="pre">false</span></code>
otherwise. No sign interpretation is necessary or performed.</li>
<li><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">uge</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ult</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ule</span></code>: interprets the operands as unsigned values and yields
<code class="docutils literal notranslate"><span class="pre">true</span></code> if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">sgt</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">sge</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">slt</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">sle</span></code>: interprets the operands as signed values and yields <code class="docutils literal notranslate"><span class="pre">true</span></code>
if <code class="docutils literal notranslate"><span class="pre">op1</span></code> is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
</ol>
<p>If the operands are <a class="reference internal" href="#t-pointer"><span class="std std-ref">pointer</span></a> typed, the pointer values
are compared as if they were integers.</p>
<p>If the operands are integer vectors, then they are compared element by
element. The result is an <code class="docutils literal notranslate"><span class="pre">i1</span></code> vector with the same number of elements
as the values being compared. Otherwise, the result is an <code class="docutils literal notranslate"><span class="pre">i1</span></code>.</p>
</div>
<div class="section" id="id305">
<h5><a class="toc-backref" href="#id2092">Example:</a><a class="headerlink" href="#id305" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = icmp eq i32 4, 5          ; yields: result=false
&lt;result&gt; = icmp ne float* %X, %X     ; yields: result=false
&lt;result&gt; = icmp ult i16  4, 5        ; yields: result=true
&lt;result&gt; = icmp sgt i16  4, 5        ; yields: result=false
&lt;result&gt; = icmp ule i16 -4, 5        ; yields: result=false
&lt;result&gt; = icmp sge i16  4, 5        ; yields: result=false
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘icmp’ instruction takes three operands. The first operand isthe condition code indicating the kind of comparison to perform. It is
not a value, just a keyword. The possible condition codes are:`,
            };
        case 'FCMP':
            return {
                url: `https://llvm.org/docs/LangRef.html#fcmp-instruction`,
                html: `<span id="i-fcmp"></span><h4><a class="toc-backref" href="#id2093">‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ Instruction</a><a class="headerlink" href="#fcmp-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id306">
<h5><a class="toc-backref" href="#id2094">Syntax:</a><a class="headerlink" href="#id306" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">fcmp</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span><span class="o">*</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">op1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">op2</span><span class="o">&gt;</span>     <span class="p">;</span> <span class="n">yields</span> <span class="n">i1</span> <span class="ow">or</span> <span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id307">
<h5><a class="toc-backref" href="#id2095">Overview:</a><a class="headerlink" href="#id307" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction returns a boolean value or vector of boolean
values based on comparison of its operands.</p>
<p>If the operands are floating-point scalars, then the result type is a
boolean (<a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a>).</p>
<p>If the operands are floating-point vectors, then the result type is a
vector of boolean with the same number of elements as the operands being
compared.</p>
</div>
<div class="section" id="id308">
<h5><a class="toc-backref" href="#id2096">Arguments:</a><a class="headerlink" href="#id308" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction takes three operands. The first operand is
the condition code indicating the kind of comparison to perform. It is
not a value, just a keyword. The possible condition codes are:</p>
<ol class="arabic simple">
<li><code class="docutils literal notranslate"><span class="pre">false</span></code>: no comparison, always returns false</li>
<li><code class="docutils literal notranslate"><span class="pre">oeq</span></code>: ordered and equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ogt</span></code>: ordered and greater than</li>
<li><code class="docutils literal notranslate"><span class="pre">oge</span></code>: ordered and greater than or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">olt</span></code>: ordered and less than</li>
<li><code class="docutils literal notranslate"><span class="pre">ole</span></code>: ordered and less than or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">one</span></code>: ordered and not equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ord</span></code>: ordered (no nans)</li>
<li><code class="docutils literal notranslate"><span class="pre">ueq</span></code>: unordered or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: unordered or greater than</li>
<li><code class="docutils literal notranslate"><span class="pre">uge</span></code>: unordered or greater than or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">ult</span></code>: unordered or less than</li>
<li><code class="docutils literal notranslate"><span class="pre">ule</span></code>: unordered or less than or equal</li>
<li><code class="docutils literal notranslate"><span class="pre">une</span></code>: unordered or not equal</li>
<li><code class="docutils literal notranslate"><span class="pre">uno</span></code>: unordered (either nans)</li>
<li><code class="docutils literal notranslate"><span class="pre">true</span></code>: no comparison, always returns true</li>
</ol>
<p><em>Ordered</em> means that neither operand is a QNAN while <em>unordered</em> means
that either operand may be a QNAN.</p>
<p>Each of <code class="docutils literal notranslate"><span class="pre">val1</span></code> and <code class="docutils literal notranslate"><span class="pre">val2</span></code> arguments must be either a <a class="reference internal" href="#t-floating"><span class="std std-ref">floating-point</span></a> type or a <a class="reference internal" href="#t-vector"><span class="std std-ref">vector</span></a> of floating-point type.
They must have identical types.</p>
</div>
<div class="section" id="id309">
<h5><a class="toc-backref" href="#id2097">Semantics:</a><a class="headerlink" href="#id309" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">fcmp</span></code>’ instruction compares <code class="docutils literal notranslate"><span class="pre">op1</span></code> and <code class="docutils literal notranslate"><span class="pre">op2</span></code> according to the
condition code given as <code class="docutils literal notranslate"><span class="pre">cond</span></code>. If the operands are vectors, then the
vectors are compared element by element. Each comparison performed
always yields an <a class="reference internal" href="#t-integer"><span class="std std-ref">i1</span></a> result, as follows:</p>
<ol class="arabic simple">
<li><code class="docutils literal notranslate"><span class="pre">false</span></code>: always yields <code class="docutils literal notranslate"><span class="pre">false</span></code>, regardless of operands.</li>
<li><code class="docutils literal notranslate"><span class="pre">oeq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ogt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">oge</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">olt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ole</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">one</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN and <code class="docutils literal notranslate"><span class="pre">op1</span></code>
is not equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ord</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if both operands are not a QNAN.</li>
<li><code class="docutils literal notranslate"><span class="pre">ueq</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ugt</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
greater than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">uge</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
greater than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ult</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
less than <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">ule</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
less than or equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">une</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN or <code class="docutils literal notranslate"><span class="pre">op1</span></code> is
not equal to <code class="docutils literal notranslate"><span class="pre">op2</span></code>.</li>
<li><code class="docutils literal notranslate"><span class="pre">uno</span></code>: yields <code class="docutils literal notranslate"><span class="pre">true</span></code> if either operand is a QNAN.</li>
<li><code class="docutils literal notranslate"><span class="pre">true</span></code>: always yields <code class="docutils literal notranslate"><span class="pre">true</span></code>, regardless of operands.</li>
</ol>
<p>The <code class="docutils literal notranslate"><span class="pre">fcmp</span></code> instruction can also optionally take any number of
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>, which are optimization hints to enable
otherwise unsafe floating-point optimizations.</p>
<p>Any set of fast-math flags are legal on an <code class="docutils literal notranslate"><span class="pre">fcmp</span></code> instruction, but the
only flags that have any effect on its semantics are those that allow
assumptions to be made about the values of input arguments; namely
<code class="docutils literal notranslate"><span class="pre">nnan</span></code>, <code class="docutils literal notranslate"><span class="pre">ninf</span></code>, and <code class="docutils literal notranslate"><span class="pre">reassoc</span></code>. See <a class="reference internal" href="#fastmath"><span class="std std-ref">Fast-Math Flags</span></a> for more information.</p>
</div>
<div class="section" id="id310">
<h5><a class="toc-backref" href="#id2098">Example:</a><a class="headerlink" href="#id310" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>&lt;result&gt; = fcmp oeq float 4.0, 5.0    ; yields: result=false
&lt;result&gt; = fcmp one float 4.0, 5.0    ; yields: result=true
&lt;result&gt; = fcmp olt float 4.0, 5.0    ; yields: result=true
&lt;result&gt; = fcmp ueq double 1.0, 2.0   ; yields: result=false
</pre></div>
</div>
</div>
`,
                tooltip: `If the operands are floating-point scalars, then the result type is aboolean (i1).`,
            };
        case 'PHI':
            return {
                url: `https://llvm.org/docs/LangRef.html#phi-instruction`,
                html: `<span id="i-phi"></span><h4><a class="toc-backref" href="#id2099">‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ Instruction</a><a class="headerlink" href="#phi-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id311">
<h5><a class="toc-backref" href="#id2100">Syntax:</a><a class="headerlink" href="#id311" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">phi</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span><span class="o">-</span><span class="n">flags</span><span class="p">]</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="p">[</span> <span class="o">&lt;</span><span class="n">val0</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">label0</span><span class="o">&gt;</span><span class="p">],</span> <span class="o">...</span>
</pre></div>
</div>
</div>
<div class="section" id="id312">
<h5><a class="toc-backref" href="#id2101">Overview:</a><a class="headerlink" href="#id312" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction is used to implement the φ node in the SSA
graph representing the function.</p>
</div>
<div class="section" id="id313">
<h5><a class="toc-backref" href="#id2102">Arguments:</a><a class="headerlink" href="#id313" title="Permalink to this headline">¶</a></h5>
<p>The type of the incoming values is specified with the first type field.
After this, the ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction takes a list of pairs as
arguments, with one pair for each predecessor basic block of the current
block. Only values of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type may be used as
the value arguments to the PHI node. Only labels may be used as the
label arguments.</p>
<p>There must be no non-phi instructions between the start of a basic block
and the PHI instructions: i.e. PHI instructions must be first in a basic
block.</p>
<p>For the purposes of the SSA form, the use of each incoming value is
deemed to occur on the edge from the corresponding predecessor block to
the current block (but after any definition of an ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’
instruction’s return value on the same edge).</p>
<p>The optional <code class="docutils literal notranslate"><span class="pre">fast-math-flags</span></code> marker indicates that the phi has one
or more <a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math-flags</span></a>. These are optimization hints
to enable otherwise unsafe floating-point optimizations. Fast-math-flags
are only valid for phis that return a floating-point scalar or vector
type, or an array (nested to any depth) of floating-point scalar or vector
types.</p>
</div>
<div class="section" id="id314">
<h5><a class="toc-backref" href="#id2103">Semantics:</a><a class="headerlink" href="#id314" title="Permalink to this headline">¶</a></h5>
<p>At runtime, the ‘<code class="docutils literal notranslate"><span class="pre">phi</span></code>’ instruction logically takes on the value
specified by the pair corresponding to the predecessor basic block that
executed just prior to the current block.</p>
</div>
<div class="section" id="id315">
<h5><a class="toc-backref" href="#id2104">Example:</a><a class="headerlink" href="#id315" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nl">Loop:</span>       <span class="c">; Infinite loop that counts from 0 on up...</span>
  <span class="nv">%indvar</span> <span class="p">=</span> <span class="k">phi</span> <span class="kt">i32</span> <span class="p">[</span> <span class="m">0</span><span class="p">,</span> <span class="nv">%LoopHeader</span> <span class="p">],</span> <span class="p">[</span> <span class="nv">%nextindvar</span><span class="p">,</span> <span class="nv">%Loop</span> <span class="p">]</span>
  <span class="nv">%nextindvar</span> <span class="p">=</span> <span class="k">add</span> <span class="kt">i32</span> <span class="nv">%indvar</span><span class="p">,</span> <span class="m">1</span>
  <span class="k">br</span> <span class="kt">label</span> <span class="nv">%Loop</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The type of the incoming values is specified with the first type field.After this, the ‘phi’ instruction takes a list of pairs as
arguments, with one pair for each predecessor basic block of the current
block. Only values of first class type may be used as
the value arguments to the PHI node. Only labels may be used as the
label arguments.`,
            };
        case 'SELECT':
            return {
                url: `https://llvm.org/docs/LangRef.html#select-instruction`,
                html: `<span id="i-select"></span><h4><a class="toc-backref" href="#id2105">‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ Instruction</a><a class="headerlink" href="#select-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id316">
<h5><a class="toc-backref" href="#id2106">Syntax:</a><a class="headerlink" href="#id316" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">select</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span> <span class="n">selty</span> <span class="o">&lt;</span><span class="n">cond</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val1</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">val2</span><span class="o">&gt;</span>             <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span>

<span class="n">selty</span> <span class="ow">is</span> <span class="n">either</span> <span class="n">i1</span> <span class="ow">or</span> <span class="p">{</span><span class="o">&lt;</span><span class="n">N</span> <span class="n">x</span> <span class="n">i1</span><span class="o">&gt;</span><span class="p">}</span>
</pre></div>
</div>
</div>
<div class="section" id="id317">
<h5><a class="toc-backref" href="#id2107">Overview:</a><a class="headerlink" href="#id317" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ instruction is used to choose one value based on a
condition, without IR-level branching.</p>
</div>
<div class="section" id="id318">
<h5><a class="toc-backref" href="#id2108">Arguments:</a><a class="headerlink" href="#id318" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">select</span></code>’ instruction requires an ‘i1’ value or a vector of ‘i1’
values indicating the condition, and two values of the same <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first
class</span></a> type.</p>
<ol class="arabic simple">
<li>The optional <code class="docutils literal notranslate"><span class="pre">fast-math</span> <span class="pre">flags</span></code> marker indicates that the select has one or more
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>. These are optimization hints to enable
otherwise unsafe floating-point optimizations. Fast-math flags are only valid
for selects that return a floating-point scalar or vector type, or an array
(nested to any depth) of floating-point scalar or vector types.</li>
</ol>
</div>
<div class="section" id="id319">
<h5><a class="toc-backref" href="#id2109">Semantics:</a><a class="headerlink" href="#id319" title="Permalink to this headline">¶</a></h5>
<p>If the condition is an i1 and it evaluates to 1, the instruction returns
the first value argument; otherwise, it returns the second value
argument.</p>
<p>If the condition is a vector of i1, then the value arguments must be
vectors of the same size, and the selection is done element by element.</p>
<p>If the condition is an i1 and the value arguments are vectors of the
same size, then an entire vector is selected.</p>
</div>
<div class="section" id="id320">
<h5><a class="toc-backref" href="#id2110">Example:</a><a class="headerlink" href="#id320" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%X</span> <span class="p">=</span> <span class="k">select</span> <span class="kt">i1</span> <span class="k">true</span><span class="p">,</span> <span class="kt">i8</span> <span class="m">17</span><span class="p">,</span> <span class="kt">i8</span> <span class="m">42</span>          <span class="c">; yields i8:17</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The ‘select’ instruction requires an ‘i1’ value or a vector of ‘i1’values indicating the condition, and two values of the same first
class type.`,
            };
        case 'FREEZE':
            return {
                url: `https://llvm.org/docs/LangRef.html#freeze-instruction`,
                html: `<span id="i-freeze"></span><h4><a class="toc-backref" href="#id2111">‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ Instruction</a><a class="headerlink" href="#freeze-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id321">
<h5><a class="toc-backref" href="#id2112">Syntax:</a><a class="headerlink" href="#id321" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">freeze</span> <span class="n">ty</span> <span class="o">&lt;</span><span class="n">val</span><span class="o">&gt;</span>    <span class="p">;</span> <span class="n">yields</span> <span class="n">ty</span><span class="p">:</span><span class="n">result</span>
</pre></div>
</div>
</div>
<div class="section" id="id322">
<h5><a class="toc-backref" href="#id2113">Overview:</a><a class="headerlink" href="#id322" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ instruction is used to stop propagation of
<a class="reference internal" href="#undefvalues"><span class="std std-ref">undef</span></a> and <a class="reference internal" href="#poisonvalues"><span class="std std-ref">poison</span></a> values.</p>
</div>
<div class="section" id="id323">
<h5><a class="toc-backref" href="#id2114">Arguments:</a><a class="headerlink" href="#id323" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">freeze</span></code>’ instruction takes a single argument.</p>
</div>
<div class="section" id="id324">
<h5><a class="toc-backref" href="#id2115">Semantics:</a><a class="headerlink" href="#id324" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id325">
<h5><a class="toc-backref" href="#id2116">Example:</a><a class="headerlink" href="#id325" title="Permalink to this headline">¶</a></h5>
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
</div>
`,
                tooltip: `The ‘freeze’ instruction takes a single argument.`,
            };
        case 'CALL':
            return {
                url: `https://llvm.org/docs/LangRef.html#call-instruction`,
                html: `<span id="i-call"></span><h4><a class="toc-backref" href="#id2117">‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ Instruction</a><a class="headerlink" href="#call-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id326">
<h5><a class="toc-backref" href="#id2118">Syntax:</a><a class="headerlink" href="#id326" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">result</span><span class="o">&gt;</span> <span class="o">=</span> <span class="p">[</span><span class="n">tail</span> <span class="o">|</span> <span class="n">musttail</span> <span class="o">|</span> <span class="n">notail</span> <span class="p">]</span> <span class="n">call</span> <span class="p">[</span><span class="n">fast</span><span class="o">-</span><span class="n">math</span> <span class="n">flags</span><span class="p">]</span> <span class="p">[</span><span class="n">cconv</span><span class="p">]</span> <span class="p">[</span><span class="n">ret</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span><span class="n">addrspace</span><span class="p">(</span><span class="o">&lt;</span><span class="n">num</span><span class="o">&gt;</span><span class="p">)]</span>
           <span class="o">&lt;</span><span class="n">ty</span><span class="o">&gt;|&lt;</span><span class="n">fnty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">fnptrval</span><span class="o">&gt;</span><span class="p">(</span><span class="o">&lt;</span><span class="n">function</span> <span class="n">args</span><span class="o">&gt;</span><span class="p">)</span> <span class="p">[</span><span class="n">fn</span> <span class="n">attrs</span><span class="p">]</span> <span class="p">[</span> <span class="n">operand</span> <span class="n">bundles</span> <span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id327">
<h5><a class="toc-backref" href="#id2119">Overview:</a><a class="headerlink" href="#id327" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ instruction represents a simple function call.</p>
</div>
<div class="section" id="id328">
<h5><a class="toc-backref" href="#id2120">Arguments:</a><a class="headerlink" href="#id328" title="Permalink to this headline">¶</a></h5>
<p>This instruction requires several arguments:</p>
<ol class="arabic">
<li><p class="first">The optional <code class="docutils literal notranslate"><span class="pre">tail</span></code> and <code class="docutils literal notranslate"><span class="pre">musttail</span></code> markers indicate that the optimizers
should perform tail call optimization. The <code class="docutils literal notranslate"><span class="pre">tail</span></code> marker is a hint that
<a class="reference external" href="CodeGenerator.html#sibcallopt">can be ignored</a>. The <code class="docutils literal notranslate"><span class="pre">musttail</span></code> marker
means that the call must be tail call optimized in order for the program to
be correct. The <code class="docutils literal notranslate"><span class="pre">musttail</span></code> marker provides these guarantees:</p>
<ol class="arabic simple">
<li>The call will not cause unbounded stack growth if it is part of a
recursive cycle in the call graph.</li>
<li>Arguments with the <a class="reference internal" href="#attr-inalloca"><span class="std std-ref">inalloca</span></a> or
<a class="reference internal" href="#attr-preallocated"><span class="std std-ref">preallocated</span></a> attribute are forwarded in place.</li>
<li>If the musttail call appears in a function with the <code class="docutils literal notranslate"><span class="pre">"thunk"</span></code> attribute
and the caller and callee both have varargs, than any unprototyped
arguments in register or memory are forwarded to the callee. Similarly,
the return value of the callee is returned to the caller’s caller, even
if a void return type is in use.</li>
</ol>
<p>Both markers imply that the callee does not access allocas from the caller.
The <code class="docutils literal notranslate"><span class="pre">tail</span></code> marker additionally implies that the callee does not access
varargs from the caller. Calls marked <code class="docutils literal notranslate"><span class="pre">musttail</span></code> must obey the following
additional  rules:</p>
<ul class="simple">
<li>The call must immediately precede a <a class="reference internal" href="#i-ret"><span class="std std-ref">ret</span></a> instruction,
or a pointer bitcast followed by a ret instruction.</li>
<li>The ret instruction must return the (possibly bitcasted) value
produced by the call, undef, or void.</li>
<li>The calling conventions of the caller and callee must match.</li>
<li>The callee must be varargs iff the caller is varargs. Bitcasting a
non-varargs function to the appropriate varargs type is legal so
long as the non-varargs prefixes obey the other rules.</li>
<li>The return type must not undergo automatic conversion to an <cite>sret</cite> pointer.</li>
</ul>
</li>
</ol>
<blockquote>
<div><p>In addition, if the calling convention is not <cite>swifttailcc</cite> or <cite>tailcc</cite>:</p>
<blockquote>
<div><ul class="simple">
<li>All ABI-impacting function attributes, such as sret, byval, inreg,
returned, and inalloca, must match.</li>
<li>The caller and callee prototypes must match. Pointer types of parameters
or return types may differ in pointee type, but not in address space.</li>
</ul>
</div></blockquote>
<p>On the other hand, if the calling convention is <cite>swifttailcc</cite> or <cite>swiftcc</cite>:</p>
<blockquote>
<div><ul class="simple">
<li>Only these ABI-impacting attributes attributes are allowed: sret, byval,
swiftself, and swiftasync.</li>
<li>Prototypes are not required to match.</li>
</ul>
<p>Tail call optimization for calls marked <code class="docutils literal notranslate"><span class="pre">tail</span></code> is guaranteed to occur if
the following conditions are met:</p>
<ul class="simple">
<li>Caller and callee both have the calling convention <code class="docutils literal notranslate"><span class="pre">fastcc</span></code> or <code class="docutils literal notranslate"><span class="pre">tailcc</span></code>.</li>
<li>The call is in tail position (ret immediately follows call and ret
uses value of call or is void).</li>
<li>Option <code class="docutils literal notranslate"><span class="pre">-tailcallopt</span></code> is enabled,
<code class="docutils literal notranslate"><span class="pre">llvm::GuaranteedTailCallOpt</span></code> is <code class="docutils literal notranslate"><span class="pre">true</span></code>, or the calling convention
is <code class="docutils literal notranslate"><span class="pre">tailcc</span></code></li>
<li><a class="reference external" href="CodeGenerator.html#tailcallopt">Platform-specific constraints are
met.</a></li>
</ul>
</div></blockquote>
</div></blockquote>
<ol class="arabic simple">
<li>The optional <code class="docutils literal notranslate"><span class="pre">notail</span></code> marker indicates that the optimizers should not add
<code class="docutils literal notranslate"><span class="pre">tail</span></code> or <code class="docutils literal notranslate"><span class="pre">musttail</span></code> markers to the call. It is used to prevent tail
call optimization from being performed on the call.</li>
<li>The optional <code class="docutils literal notranslate"><span class="pre">fast-math</span> <span class="pre">flags</span></code> marker indicates that the call has one or more
<a class="reference internal" href="#fastmath"><span class="std std-ref">fast-math flags</span></a>, which are optimization hints to enable
otherwise unsafe floating-point optimizations. Fast-math flags are only valid
for calls that return a floating-point scalar or vector type, or an array
(nested to any depth) of floating-point scalar or vector types.</li>
<li>The optional “cconv” marker indicates which <a class="reference internal" href="#callingconv"><span class="std std-ref">calling
convention</span></a> the call should use. If none is
specified, the call defaults to using C calling conventions. The
calling convention of the call must match the calling convention of
the target function, or else the behavior is undefined.</li>
<li>The optional <a class="reference internal" href="#paramattrs"><span class="std std-ref">Parameter Attributes</span></a> list for return
values. Only ‘<code class="docutils literal notranslate"><span class="pre">zeroext</span></code>’, ‘<code class="docutils literal notranslate"><span class="pre">signext</span></code>’, and ‘<code class="docutils literal notranslate"><span class="pre">inreg</span></code>’ attributes
are valid here.</li>
<li>The optional addrspace attribute can be used to indicate the address space
of the called function. If it is not specified, the program address space
from the <a class="reference internal" href="#langref-datalayout"><span class="std std-ref">datalayout string</span></a> will be used.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">ty</span></code>’: the type of the call instruction itself which is also the
type of the return value. Functions that return no value are marked
<code class="docutils literal notranslate"><span class="pre">void</span></code>.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnty</span></code>’: shall be the signature of the function being called. The
argument types must match the types implied by this signature. This
type can be omitted if the function is not varargs.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">fnptrval</span></code>’: An LLVM value containing a pointer to a function to
be called. In most cases, this is a direct function call, but
indirect <code class="docutils literal notranslate"><span class="pre">call</span></code>’s are just as possible, calling an arbitrary pointer
to function value.</li>
<li>‘<code class="docutils literal notranslate"><span class="pre">function</span> <span class="pre">args</span></code>’: argument list whose types match the function
signature argument types and parameter attributes. All arguments must
be of <a class="reference internal" href="#t-firstclass"><span class="std std-ref">first class</span></a> type. If the function signature
indicates the function accepts a variable number of arguments, the
extra arguments can be specified.</li>
<li>The optional <a class="reference internal" href="#fnattrs"><span class="std std-ref">function attributes</span></a> list.</li>
<li>The optional <a class="reference internal" href="#opbundles"><span class="std std-ref">operand bundles</span></a> list.</li>
</ol>
</div>
<div class="section" id="id329">
<h5><a class="toc-backref" href="#id2121">Semantics:</a><a class="headerlink" href="#id329" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">call</span></code>’ instruction is used to cause control flow to transfer to
a specified function, with its incoming arguments bound to the specified
values. Upon a ‘<code class="docutils literal notranslate"><span class="pre">ret</span></code>’ instruction in the called function, control
flow continues with the instruction after the function call, and the
return value of the function is bound to the result argument.</p>
</div>
<div class="section" id="id330">
<h5><a class="toc-backref" href="#id2122">Example:</a><a class="headerlink" href="#id330" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="nv">%retval</span> <span class="p">=</span> <span class="k">call</span> <span class="kt">i32</span> <span class="vg">@test</span><span class="p">(</span><span class="kt">i32</span> <span class="nv">%argc</span><span class="p">)</span>
<span class="k">call</span> <span class="kt">i32</span> <span class="p">(</span><span class="kt">i8</span><span class="p">*,</span> <span class="p">...)*</span> <span class="vg">@printf</span><span class="p">(</span><span class="kt">i8</span><span class="p">*</span> <span class="nv">%msg</span><span class="p">,</span> <span class="kt">i32</span> <span class="m">12</span><span class="p">,</span> <span class="kt">i8</span> <span class="m">42</span><span class="p">)</span>        <span class="c">; yields i32</span>
<span class="nv">%X</span> <span class="p">=</span> <span class="k">tail</span> <span class="k">call</span> <span class="kt">i32</span> <span class="vg">@foo</span><span class="p">()</span>                                    <span class="c">; yields i32</span>
<span class="nv">%Y</span> <span class="p">=</span> <span class="k">tail</span> <span class="k">call</span> <span class="k">fastcc</span> <span class="kt">i32</span> <span class="vg">@foo</span><span class="p">()</span>  <span class="c">; yields i32</span>
<span class="k">call</span> <span class="k">void</span> <span class="nv">%foo</span><span class="p">(</span><span class="kt">i8</span> <span class="k">signext</span> <span class="m">97</span><span class="p">)</span>

<span class="nv">%struct.A</span> <span class="p">=</span> <span class="k">type</span> <span class="p">{</span> <span class="kt">i32</span><span class="p">,</span> <span class="kt">i8</span> <span class="p">}</span>
<span class="nv">%r</span> <span class="p">=</span> <span class="k">call</span> <span class="nv">%struct.A</span> <span class="vg">@foo</span><span class="p">()</span>                        <span class="c">; yields { i32, i8 }</span>
<span class="nv">%gr</span> <span class="p">=</span> <span class="k">extractvalue</span> <span class="nv">%struct.A</span> <span class="nv">%r</span><span class="p">,</span> <span class="m">0</span>                <span class="c">; yields i32</span>
<span class="nv">%gr1</span> <span class="p">=</span> <span class="k">extractvalue</span> <span class="nv">%struct.A</span> <span class="nv">%r</span><span class="p">,</span> <span class="m">1</span>               <span class="c">; yields i8</span>
<span class="nv">%Z</span> <span class="p">=</span> <span class="k">call</span> <span class="k">void</span> <span class="vg">@foo</span><span class="p">()</span> <span class="k">noreturn</span>                    <span class="c">; indicates that %foo never returns normally</span>
<span class="nv">%ZZ</span> <span class="p">=</span> <span class="k">call</span> <span class="k">zeroext</span> <span class="kt">i32</span> <span class="vg">@bar</span><span class="p">()</span>                     <span class="c">; Return value is %zero extended</span>
</pre></div>
</div>
<p>llvm treats calls to some functions with names and arguments that match
the standard C99 library as being the C99 library functions, and may
perform optimizations or generate code for them under that assumption.
This is something we’d like to change in the future to provide better
support for freestanding environments and non-C-based languages.</p>
</div>
`,
                tooltip: `This instruction requires several arguments:`,
            };
        case 'VA-ARG':
            return {
                url: `https://llvm.org/docs/LangRef.html#va-arg-instruction`,
                html: `<span id="i-va-arg"></span><h4><a class="toc-backref" href="#id2123">‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ Instruction</a><a class="headerlink" href="#va-arg-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id331">
<h5><a class="toc-backref" href="#id2124">Syntax:</a><a class="headerlink" href="#id331" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">va_arg</span> <span class="o">&lt;</span><span class="n">va_list</span><span class="o">*&gt;</span> <span class="o">&lt;</span><span class="n">arglist</span><span class="o">&gt;</span><span class="p">,</span> <span class="o">&lt;</span><span class="n">argty</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id332">
<h5><a class="toc-backref" href="#id2125">Overview:</a><a class="headerlink" href="#id332" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ instruction is used to access arguments passed through
the “variable argument” area of a function call. It is used to implement
the <code class="docutils literal notranslate"><span class="pre">va_arg</span></code> macro in C.</p>
</div>
<div class="section" id="id333">
<h5><a class="toc-backref" href="#id2126">Arguments:</a><a class="headerlink" href="#id333" title="Permalink to this headline">¶</a></h5>
<p>This instruction takes a <code class="docutils literal notranslate"><span class="pre">va_list*</span></code> value and the type of the
argument. It returns a value of the specified argument type and
increments the <code class="docutils literal notranslate"><span class="pre">va_list</span></code> to point to the next argument. The actual
type of <code class="docutils literal notranslate"><span class="pre">va_list</span></code> is target specific.</p>
</div>
<div class="section" id="id334">
<h5><a class="toc-backref" href="#id2127">Semantics:</a><a class="headerlink" href="#id334" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">va_arg</span></code>’ instruction loads an argument of the specified type
from the specified <code class="docutils literal notranslate"><span class="pre">va_list</span></code> and causes the <code class="docutils literal notranslate"><span class="pre">va_list</span></code> to point to
the next argument. For more information, see the variable argument
handling <a class="reference internal" href="#int-varargs"><span class="std std-ref">Intrinsic Functions</span></a>.</p>
<p>It is legal for this instruction to be called in a function which does
not take a variable number of arguments, for example, the <code class="docutils literal notranslate"><span class="pre">vfprintf</span></code>
function.</p>
<p><code class="docutils literal notranslate"><span class="pre">va_arg</span></code> is an LLVM instruction instead of an <a class="reference internal" href="#intrinsics"><span class="std std-ref">intrinsic
function</span></a> because it takes a type as an argument.</p>
</div>
<div class="section" id="id335">
<h5><a class="toc-backref" href="#id2128">Example:</a><a class="headerlink" href="#id335" title="Permalink to this headline">¶</a></h5>
<p>See the <a class="reference internal" href="#int-varargs"><span class="std std-ref">variable argument processing</span></a> section.</p>
<p>Note that the code generator does not yet fully support va_arg on many
targets. Also, it does not currently support va_arg with aggregate
types on any target.</p>
</div>
`,
                tooltip: `This instruction takes a va_list* value and the type of theargument. It returns a value of the specified argument type and
increments the va_list to point to the next argument. The actual
type of va_list is target specific.`,
            };
        case 'LANDINGPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#landingpad-instruction`,
                html: `<span id="i-landingpad"></span><h4><a class="toc-backref" href="#id2129">‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ Instruction</a><a class="headerlink" href="#landingpad-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id336">
<h5><a class="toc-backref" href="#id2130">Syntax:</a><a class="headerlink" href="#id336" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">landingpad</span> <span class="o">&lt;</span><span class="n">resultty</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;+</span>
<span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">landingpad</span> <span class="o">&lt;</span><span class="n">resultty</span><span class="o">&gt;</span> <span class="n">cleanup</span> <span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;*</span>

<span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;</span> <span class="o">:=</span> <span class="n">catch</span> <span class="o">&lt;</span><span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">value</span><span class="o">&gt;</span>
<span class="o">&lt;</span><span class="n">clause</span><span class="o">&gt;</span> <span class="o">:=</span> <span class="nb">filter</span> <span class="o">&lt;</span><span class="n">array</span> <span class="n">constant</span> <span class="nb">type</span><span class="o">&gt;</span> <span class="o">&lt;</span><span class="n">array</span> <span class="n">constant</span><span class="o">&gt;</span>
</pre></div>
</div>
</div>
<div class="section" id="id337">
<h5><a class="toc-backref" href="#id2131">Overview:</a><a class="headerlink" href="#id337" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling
system</a> to specify that a basic block
is a landing pad — one where the exception lands, and corresponds to the
code found in the <code class="docutils literal notranslate"><span class="pre">catch</span></code> portion of a <code class="docutils literal notranslate"><span class="pre">try</span></code>/<code class="docutils literal notranslate"><span class="pre">catch</span></code> sequence. It
defines values supplied by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> upon
re-entry to the function. The <code class="docutils literal notranslate"><span class="pre">resultval</span></code> has the type <code class="docutils literal notranslate"><span class="pre">resultty</span></code>.</p>
</div>
<div class="section" id="id339">
<h5><a class="toc-backref" href="#id2132">Arguments:</a><a class="headerlink" href="#id339" title="Permalink to this headline">¶</a></h5>
<p>The optional
<code class="docutils literal notranslate"><span class="pre">cleanup</span></code> flag indicates that the landing pad block is a cleanup.</p>
<p>A <code class="docutils literal notranslate"><span class="pre">clause</span></code> begins with the clause type — <code class="docutils literal notranslate"><span class="pre">catch</span></code> or <code class="docutils literal notranslate"><span class="pre">filter</span></code> — and
contains the global variable representing the “type” that may be caught
or filtered respectively. Unlike the <code class="docutils literal notranslate"><span class="pre">catch</span></code> clause, the <code class="docutils literal notranslate"><span class="pre">filter</span></code>
clause takes an array constant as its argument. Use
“<code class="docutils literal notranslate"><span class="pre">[0</span> <span class="pre">x</span> <span class="pre">i8**]</span> <span class="pre">undef</span></code>” for a filter which cannot throw. The
‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction must contain <em>at least</em> one <code class="docutils literal notranslate"><span class="pre">clause</span></code> or
the <code class="docutils literal notranslate"><span class="pre">cleanup</span></code> flag.</p>
</div>
<div class="section" id="id340">
<h5><a class="toc-backref" href="#id2133">Semantics:</a><a class="headerlink" href="#id340" title="Permalink to this headline">¶</a></h5>
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
<li>A landing pad block is a basic block which is the unwind destination
of an ‘<code class="docutils literal notranslate"><span class="pre">invoke</span></code>’ instruction.</li>
<li>A landing pad block must have a ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction as its
first non-PHI instruction.</li>
<li>There can be only one ‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction within the landing
pad block.</li>
<li>A basic block that is not a landing pad block may not include a
‘<code class="docutils literal notranslate"><span class="pre">landingpad</span></code>’ instruction.</li>
</ul>
</div>
<div class="section" id="id341">
<h5><a class="toc-backref" href="#id2134">Example:</a><a class="headerlink" href="#id341" title="Permalink to this headline">¶</a></h5>
<div class="highlight-llvm notranslate"><div class="highlight"><pre><span></span><span class="c">;; A landing pad which can catch an integer.</span>
<span class="nv">%res</span> <span class="p">=</span> <span class="k">landingpad</span> <span class="p">{</span> <span class="kt">i8</span><span class="p">*,</span> <span class="kt">i32</span> <span class="p">}</span>
         <span class="k">catch</span> <span class="kt">i8</span><span class="p">**</span> <span class="vg">@_ZTIi</span>
<span class="c">;; A landing pad that is a cleanup.</span>
<span class="nv">%res</span> <span class="p">=</span> <span class="k">landingpad</span> <span class="p">{</span> <span class="kt">i8</span><span class="p">*,</span> <span class="kt">i32</span> <span class="p">}</span>
         <span class="k">cleanup</span>
<span class="c">;; A landing pad which can catch an integer and can only throw a double.</span>
<span class="nv">%res</span> <span class="p">=</span> <span class="k">landingpad</span> <span class="p">{</span> <span class="kt">i8</span><span class="p">*,</span> <span class="kt">i32</span> <span class="p">}</span>
         <span class="k">catch</span> <span class="kt">i8</span><span class="p">**</span> <span class="vg">@_ZTIi</span>
         <span class="k">filter</span> <span class="p">[</span><span class="m">1</span> <span class="p">x</span> <span class="kt">i8</span><span class="p">**]</span> <span class="p">[</span><span class="kt">i8</span><span class="p">**</span> <span class="vg">@_ZTId</span><span class="p">]</span>
</pre></div>
</div>
</div>
`,
                tooltip: `The optionalcleanup flag indicates that the landing pad block is a cleanup.`,
            };
        case 'CATCHPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#catchpad-instruction`,
                html: `<span id="i-catchpad"></span><h4><a class="toc-backref" href="#id2135">‘<code class="docutils literal notranslate"><span class="pre">catchpad</span></code>’ Instruction</a><a class="headerlink" href="#catchpad-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id342">
<h5><a class="toc-backref" href="#id2136">Syntax:</a><a class="headerlink" href="#id342" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">catchpad</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">catchswitch</span><span class="o">&gt;</span> <span class="p">[</span><span class="o">&lt;</span><span class="n">args</span><span class="o">&gt;*</span><span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id343">
<h5><a class="toc-backref" href="#id2137">Overview:</a><a class="headerlink" href="#id343" title="Permalink to this headline">¶</a></h5>
<p>The ‘<code class="docutils literal notranslate"><span class="pre">catchpad</span></code>’ instruction is used by <a class="reference external" href="ExceptionHandling.html#overview">LLVM’s exception handling
system</a> to specify that a basic block
begins a catch handler — one where a personality routine attempts to transfer
control to catch an exception.</p>
</div>
<div class="section" id="id345">
<h5><a class="toc-backref" href="#id2138">Arguments:</a><a class="headerlink" href="#id345" title="Permalink to this headline">¶</a></h5>
<p>The <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code> operand must always be a token produced by a
<a class="reference internal" href="#i-catchswitch"><span class="std std-ref">catchswitch</span></a> instruction in a predecessor block. This
ensures that each <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> has exactly one predecessor block, and it always
terminates in a <code class="docutils literal notranslate"><span class="pre">catchswitch</span></code>.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">args</span></code> correspond to whatever information the personality routine
requires to know if this is an appropriate handler for the exception. Control
will transfer to the <code class="docutils literal notranslate"><span class="pre">catchpad</span></code> if this is the first appropriate handler for
the exception.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">resultval</span></code> has the type <a class="reference internal" href="#t-token"><span class="std std-ref">token</span></a> and is used to match the
<code class="docutils literal notranslate"><span class="pre">catchpad</span></code> to corresponding <a class="reference internal" href="#i-catchret"><span class="std std-ref">catchrets</span></a> and other nested EH
pads.</p>
</div>
<div class="section" id="id346">
<h5><a class="toc-backref" href="#id2139">Semantics:</a><a class="headerlink" href="#id346" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id348">
<h5><a class="toc-backref" href="#id2140">Example:</a><a class="headerlink" href="#id348" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>dispatch:
  %cs = catchswitch within none [label %handler0] unwind to caller
  ;; A catch block which can catch an integer.
handler0:
  %tok = catchpad within %cs [i8** @_ZTIi]
</pre></div>
</div>
</div>
`,
                tooltip: `The catchswitch operand must always be a token produced by acatchswitch instruction in a predecessor block. This
ensures that each catchpad has exactly one predecessor block, and it always
terminates in a catchswitch.`,
            };
        case 'CLEANUPPAD':
            return {
                url: `https://llvm.org/docs/LangRef.html#cleanuppad-instruction`,
                html: `<span id="i-cleanuppad"></span><h4><a class="toc-backref" href="#id2141">‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ Instruction</a><a class="headerlink" href="#cleanuppad-instruction" title="Permalink to this headline">¶</a></h4>
<div class="section" id="id349">
<h5><a class="toc-backref" href="#id2142">Syntax:</a><a class="headerlink" href="#id349" title="Permalink to this headline">¶</a></h5>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="o">&lt;</span><span class="n">resultval</span><span class="o">&gt;</span> <span class="o">=</span> <span class="n">cleanuppad</span> <span class="n">within</span> <span class="o">&lt;</span><span class="n">parent</span><span class="o">&gt;</span> <span class="p">[</span><span class="o">&lt;</span><span class="n">args</span><span class="o">&gt;*</span><span class="p">]</span>
</pre></div>
</div>
</div>
<div class="section" id="id350">
<h5><a class="toc-backref" href="#id2143">Overview:</a><a class="headerlink" href="#id350" title="Permalink to this headline">¶</a></h5>
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
</div>
<div class="section" id="id352">
<h5><a class="toc-backref" href="#id2144">Arguments:</a><a class="headerlink" href="#id352" title="Permalink to this headline">¶</a></h5>
<p>The instruction takes a list of arbitrary values which are interpreted
by the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a>.</p>
</div>
<div class="section" id="id353">
<h5><a class="toc-backref" href="#id2145">Semantics:</a><a class="headerlink" href="#id353" title="Permalink to this headline">¶</a></h5>
<p>When the call stack is being unwound due to an exception being thrown,
the <a class="reference internal" href="#personalityfn"><span class="std std-ref">personality function</span></a> transfers control to the
<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> with the aid of the personality-specific arguments.
As with calling conventions, how the personality function results are
represented in LLVM IR is target specific.</p>
<p>The <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> instruction has several restrictions:</p>
<ul class="simple">
<li>A cleanup block is a basic block which is the unwind destination of
an exceptional instruction.</li>
<li>A cleanup block must have a ‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction as its
first non-PHI instruction.</li>
<li>There can be only one ‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction within the
cleanup block.</li>
<li>A basic block that is not a cleanup block may not include a
‘<code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code>’ instruction.</li>
</ul>
<p>When a <code class="docutils literal notranslate"><span class="pre">cleanuppad</span></code> has been “entered” but not yet “exited” (as
described in the <a class="reference external" href="ExceptionHandling.html#wineh-constraints">EH documentation</a>),
it is undefined behavior to execute a <a class="reference internal" href="#i-call"><span class="std std-ref">call</span></a> or <a class="reference internal" href="#i-invoke"><span class="std std-ref">invoke</span></a>
that does not carry an appropriate <a class="reference internal" href="#ob-funclet"><span class="std std-ref">“funclet” bundle</span></a>.</p>
</div>
<div class="section" id="id355">
<h5><a class="toc-backref" href="#id2146">Example:</a><a class="headerlink" href="#id355" title="Permalink to this headline">¶</a></h5>
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>%tok = cleanuppad within %cs []
</pre></div>
</div>
</div>
`,
                tooltip: `The instruction takes a list of arbitrary values which are interpretedby the personality function.`,
            };
    }
}
