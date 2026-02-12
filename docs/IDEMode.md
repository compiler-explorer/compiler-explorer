# IDE Mode (Tree View)

The "IDE Mode", also known as the "Tree View" allows you to work with multiple files.

## Features

- **File Explorer:** The Tree View displays your project's files in a hierarchical structure, similar to a traditional IDE.
- **File Operations:** You can perform various file operations directly from the Tree View:
    - **Add new files:** Create new source files.
    - **Rename files:** Change the names of existing files.
    - **Delete files:** Remove files from your project.
    - **Include/exclude files from the build:** Choose which files to include in a compilation.
- **Project Management:**
    - **Save and Load:** Save your entire project, including all files and settings, as a `.zip` archive. You can later load this `.zip` file to restore your project.
    - **CMake Support:** The Tree View supports CMake projects. You can enable CMake mode and provide custom arguments for your build.
- **Drag and Drop:** You can drag and drop files from your computer directly into the Tree View to add them to your project.
- **Language Support:** The Tree View is language-aware and allows you to select the primary language for your project.

## How to Access

You can open the Tree View by clicking on the "Add..." button in the top menu and selecting "Tree (IDE Mode)".

## Implementation Details

The core logic for the Tree View is implemented in the `static/panes/tree.ts` file. This file handles the UI, file operations, and communication with other parts of Compiler Explorer, such as the editors and compilers. The UI template for the tree view is in `views/templates/panes/tree.pug`.
