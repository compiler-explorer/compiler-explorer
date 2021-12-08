# Adding Assembly Documentation for a new instruction set

This document explains how to add assembly documentation for a new instruction set to Compiler Explorer
("CE" from here on).

If you were not already aware, CE has both quick-tip and more thorough assembly instruction documentation available for
a couple instruction sets (currently JVM bytecode, amd64 and arm32). The feature is demonstrated in the gif below.

![Demo of Assembly Documentation](images/show_assembly_documentation.gif)

To add a new assembly documentation handler, you need to perform the following steps:

## 1. Find a data source

First of all you need to find a data source to get our instruction info from. While it is possible to write down
information about every single instruction for an instruction set, it's far from maintainable and it is a lot of work.

Existing assembly documentation handlers use some sort of established documentation. The arm32 handler uses the
developer.arm.com website and the JVM bytecode handler uses Oracle's documentation.

## 2. Create a tool for collecting the data

Since we want to go through the automated route, you should write a script or a piece of code to automatically gather the
data for us and store it in a nice format that CE expects. The output of the script should be a generated .js file
with a single exported function containing a gigantic switch for the instruction opcode. Examples of this generated file
can be found in `/lib/handlers/asm-docs-amd64.js`.

How you generate this file is completely up to you, just make sure it's easy for others to run the script if needed as
well. If you need inspiration on how to write this tool, you can look at the `docenizer-*` scripts found in
`/etc/scripts` in the source control tree.

CE expects the tool to output the file into the `/lib/handlers/` folder with a name following the existing convention.
Each case in the switch should return a piece of formatted HTML to insert into the popup, a tooltip text for the
on-hover tooltip and a URL to external documentation.

```js
case "CALL":
    return {
        "html": "[html to embed into the popup]",
        "tooltip": "[text to show in the hover tooltip]",
        "url": "http://www.felixcloutier.com/x86/CALL.html"
    };
```

## 3. Connect your tool output to CE

Once your tool has generated the JavaScript file, you want to connect it to CE. This is done by editing the files found
in `/lib/handlers/assembly-documentation`. You'll want to add a new file named after your instruction set which contains
a class extending `BaseAssemblyDocumentationHandler`. The class should implement the `getInstructionInformation` method.

This method is expected to take the instruction opcode in full uppercase and either return the associated data or null
if not found.

```js
import { getAsmOpcode } from '../asm-docs-java';
import { BaseAssemblyDocumentationHandler } from '../base-assembly-documentation-handler';

export class JavaDocumentationHandler extends BaseAssemblyDocumentationHandler {
    getInstructionInformation(instruction) {
        return getAsmOpcode(instruction) || null;
    }
}
```

The last thing to do is to associate your instruction set in the CE API with this handler. This is done by modifying the
`/lib/handlers/assembly-documentation/router.js` file. Simply add your instruction set name and associate it with a
new instance of your class in the mapping object.

```js
const ASSEMBLY_DOCUMENTATION_HANDLERS = {
    amd64: new Amd64DocumentationHandler(),
    arm32: new Arm32DocumentationHandler(),
    java: new JavaDocumentationHandler(),
};
```

## 4. Testing

You can ensure your API handler is working as expected by writing a test case for it in the
`/test/handlers/assembly-documentation` directory. Simply copy over one of the existing files and modify it to work
with the new slug you created in the step above.
