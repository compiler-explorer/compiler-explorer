.class LLSLqLuLaLrLeL;
.super Ljava/lang/Object;
.source "example.java"


# direct methods
.method constructor <init>()V
    .registers 1

    #@0
    .line 12
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    #@3
    return-void
.end method

.method static square(I)I
    .registers 1

    #@0
    .line 14
    mul-int/2addr p0, p0

    #@1
    return p0
.end method
