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
    - **CMake Support:** The Tree View supports CMake projects. You can enable CMake mode and provide custom arguments for your build. Note that you will need a file called `CMakeLists.txt` in your project for this to work. CMake mode is only available when the selected language is C, C++, Fortran, or CUDA.
- **Drag and Drop:** You can drag and drop files from your computer directly into the Tree View to add them to your project.
- **Language Support:** The Tree View is language-aware and allows you to select the primary language for your project.

## Ctrl+S Behaviour

When a Tree View is active, pressing **Ctrl+S** (or **Cmd+S** on macOS) in any editor will include that editor's file in the tree and refresh it. If the file has not yet been named, a dialog will appear prompting you to enter a filename before it is included. This is controlled by the **"In Tree (IDE) mode, Ctrl+S includes the current file"** setting, which is enabled by default.

If the setting is disabled, or if there is no active Tree View, Ctrl+S falls back to the behaviour configured in the general **"Ctrl+S"** setting:

- **Save To Local File** (default) — downloads the editor content as a file.
- **Create Short Link** — generates and copies a short link.
- **Reformat code** — runs the code formatter.
- **Do nothing** — ignores the keypress.

## Tab Behaviour

When a Tree View is active, opening files from the tree will add their editors as tabs in the same editor panel rather than creating separate panels. Clicking a file that is already open will switch to its existing tab instead of opening a duplicate. New panes added from the top menu (such as editors or diff views) are also stacked into the editor area rather than being placed at the root of the layout.

## Adding Compilers and Executors

> **Important:** In IDE mode, compilers and executors must be added from the Tree View panel using its **"Add compiler"** and **"Add executor"** buttons. Do **not** add compilers from an individual source editor's dropdown — those are not associated with the tree and will not participate in multi-file builds. Only compilers and executors that belong to the tree can see all the files managed by it.

## How to Access

You can open the Tree View by clicking on the "Add..." button in the top menu and selecting "Tree (IDE Mode)".

## Implementation Details

The core logic for the Tree View is implemented in the `static/panes/tree.ts` file. This file handles the UI, file operations, and communication with other parts of Compiler Explorer, such as the editors and compilers. The UI template for the tree view is in `views/templates/panes/tree.pug`.
