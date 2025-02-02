// For more information on using R8 and keep annotations, please refer to
// - https://r8.googlesource.com/r8
// - https://r8.googlesource.com/r8/+/refs/heads/main/doc/keepanno-guide.md

import com.android.tools.r8.keepanno.annotations.KeepItemKind
import com.android.tools.r8.keepanno.annotations.UsedByReflection

@UsedByReflection(kind = KeepItemKind.CLASS_AND_MEMBERS)
fun main(args: Array<String>) {
    // code...
}
