// Type your code here, or load an example.
class Square {
    static int square(int num) {
        return num * num;
    }
}

// Set `enabled` to `true` to enable profile-guided compilation for dex2oat.
// For the profile format, see
// https://developer.android.com/topic/performance/baselineprofiles/manually-create-measure#define-rules-manually
// (For advanced use only!)
/* ---------- begin profile (enabled=false) ----------
HSPLSquare;-><init>()V
HSPLSquare;->square(I)I
LSquare;
---------- end profile ---------- */
