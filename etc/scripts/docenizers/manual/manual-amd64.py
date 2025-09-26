manual_instructions_amd64 = [
    {
        "names": ["LWPINS"],
        "tooltip": "Inserts a custom event record into the Lightweight Profiling (LWP) event ring buffer, allowing programs to log significant events like library loads or thread state changes. If the buffer is full, the oldest record is overwritten.",
        "url": "https://docs.amd.com/v/u/en-US/24594_3.37#page=265",
        "html": """<p>Lightweight Profiling Insert Record. Part of AMD's Lightweight Profiling (LWP) extension.</p>
              <p>Inserts a programmed event record into the LWP event ring buffer in memory and advances the ring buffer pointer. The record has an EventId of 255.</p>
              <p><b>Operands:</b> 
              <ul>
              <li>First operand (register): Stored in the Data2 field (bytes 23-16)</li>
              <li>Second operand (register/memory): Stored in the Data1 field (bytes 7-4)</li>
              <li>Third operand (immediate): Truncated to 16 bits and stored in the Flags field (bytes 3-2)</li>
              </ul>
              </p>
              <p>If the ring buffer is not full or LWP is in Continuous Mode, the head pointer is advanced and CF is cleared. If the buffer is full in Synchronized Mode, the last record is overwritten, MissedEvents counter is incremented, head pointer is not advanced, and CF is set.</p>
              <p>This instruction allows programs to mark significant events (e.g., library loads/unloads, thread state changes) in the profiling buffer.</p>
              <p>Generates #UD if not in protected mode or LWP not available. Clears CF harmlessly if LWP is disabled.</p>"""
    },
    {
        "names": ["LWPVAL"],
        "tooltip": "Decrements the event counter for a programmed value sample event. If the counter goes negative, it logs an event in the LWP event ring buffer, enabling value profiling of program variables at specified intervals.",
        "url": "https://docs.amd.com/v/u/en-US/24594_3.37#page=267",
        "html": """<p>Lightweight Profiling Insert Value. Part of AMD's Lightweight Profiling (LWP) extension.</p>
              <p>Decrements the event counter associated with the programmed value sample event. If the resulting counter value is negative, inserts an event record into the LWP event ring buffer and advances the ring buffer pointer. The record has an EventId of 1.</p>
              <p><b>Operands:</b>
              <ul>
              <li>First operand (register): Stored in the Data2 field (bytes 23-16)</li>
              <li>Second operand (register/memory): Stored in the Data1 field (bytes 7-4)</li>
              <li>Third operand (immediate): Truncated to 16 bits and stored in the Flags field (bytes 3-2)</li>
              </ul>
              </p>
              <p>If no record is written, the memory operand (if used) is not accessed. The event counter is reset to its interval if the record is written.</p>
              <p>This instruction is used for value profiling - sampling values of program variables at a predetermined frequency without modifying registers or flags.</p>
              <p>Generates #UD if not in protected mode or LWP not available. Does nothing harmlessly if LWP is disabled or value sampling is not enabled.</p>"""
    },
    {
        "names": ["SLWPCB"],
        "tooltip": "Stores the address of the Lightweight Profiling Control Block (LWPCB) and flushes the current profiling state to memory. If LWP is disabled, the specified register is set to zero.",
        "url": "https://docs.amd.com/v/u/en-US/24594_3.37#page=386",
        "html": """<p>Store Lightweight Profiling Control Block Address. Part of AMD's Lightweight Profiling (LWP) extension.</p>
              <p>Flushes LWP state to memory and returns the current effective address of the Lightweight Profiling Control Block (LWPCB) in the specified register. The LWPCB address is truncated to 32 bits if the operand size is 32.</p>
              <p>The flush operation stores internal event counters for active events and the current ring buffer head pointer into the LWPCB. If there is an unwritten event record pending, it is written to the event ring buffer.</p>
              <p>If LWP is not currently enabled, SLWPCB sets the specified register to zero. The address calculation involves subtracting the current DS.Base address from the linear address kept in LWP_CBADDR MSR.</p>
              <p>Generates #UD if not in protected mode or if LWP not available. Using SLWPCB when CPL ≠ 3 or in SMM is not recommended unless LWPCB and ring buffer are properly mapped.</p>""",
    },
    {
        "names": ["LLWPCB"],
        "tooltip": "Loads the address of the Lightweight Profiling Control Block (LWPCB) from the specified register, enabling or disabling profiling as needed. If LWP is already enabled, it flushes the current state to memory.",
        "url": "https://docs.amd.com/v/u/en-US/24594_3.37#page=258",
        "html": """<p>Load Lightweight Profiling Control Block Address. Part of AMD's Lightweight Profiling (LWP) extension.</p>
              <p>Parses the Lightweight Profiling Control Block (LWPCB) at the address contained in the specified register. If valid, writes the address into the LWP_CBADDR MSR and enables Lightweight Profiling.</p>
              <p>If LWP is already enabled, the processor first flushes the current LWP state to memory in the old LWPCB (similar to SLWPCB). If the specified LWPCB address is zero, LWP is disabled.</p>
              <p>Validates the LWPCB address and ring buffer parameters - may generate #GP exceptions for programming errors or #PF for page faults during validation.</p>
              <p>Configures LWP events based on LWPCB.Flags field, setting internal counters from EventCounter values and adjusting intervals to implementation-defined minimums if necessary.</p>
              <p>Generates #UD if not in protected mode or if LWP not available. Using LLWPCB when CPL ≠ 3 or in SMM is not recommended unless both old and new LWPCBs and ring buffers are properly mapped.</p>"""
    }
]
