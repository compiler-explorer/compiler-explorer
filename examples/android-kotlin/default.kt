// Type your code here, or load an example.
//
// For R8, please load R8Example for an example with a keep annotation,
// which will prevent unused code from being removed from the final output.
fun square(num: Int): Int = num * num

// Set `enabled` to `true` to enable profile-guided compilation for dex2oat.
// For the profile format, see
// https://developer.android.com/topic/performance/baselineprofiles/manually-create-measure#define-rules-manually
// (For advanced use only!)
/* ---------- begin profile (enabled=false) ----------
HSPLExampleKt;->square(I)I
LExampleKt;
---------- end profile ---------- */
