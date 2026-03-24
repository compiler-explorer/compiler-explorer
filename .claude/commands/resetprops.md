Reset all `.local.properties` files in `etc/config/` by overwriting them with the contents of their corresponding `.amazon.properties` files.

Steps:
1. Find all `*.local.properties` files in `etc/config/`
2. For each file, determine the base name (e.g. `c++.local.properties` -> `c++`)
3. Check if a corresponding `*.amazon.properties` file exists (e.g. `c++.amazon.properties`)
4. If it exists, overwrite the `.local.properties` file with the contents of the `.amazon.properties` file
5. If no corresponding `.amazon.properties` file exists, skip it and note that it was skipped
6. Report a summary of which files were overwritten and which were skipped
