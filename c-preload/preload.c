// Copyright (c) 2012-2015, Matt Godbolt
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#ifndef O_CREAT
#define O_CREAT 0100
#endif

static int allowed_match(const char* path, const char* okpath) {
    char resolvedBuf[PATH_MAX];
    const char* resolved = path;
    if (!strncmp(resolved, "/proc/self", 10)) {
        // Leave references to /proc/self.* alone as its real path is different
        // each time.
    } else {
        resolved = realpath(path, resolvedBuf);
        if (resolved == NULL) {
            return 0;
        }
    }

    while (*okpath) {
        const char* end = strchrnul(okpath, ':');
        if (strncmp(okpath, resolved, end - okpath) == 0) return 1;
        okpath = end;
        while (*okpath == ':') ++okpath;
    }

    fprintf(stderr, "Access to \"%s\" denied by gcc-explorer policy\n", path);
    errno = EACCES;
    return 0;
}

static int allowed_env(const char* pathname, const char* envvar) {
    const char* okpath = getenv(envvar);
    if (okpath == NULL) {
       errno = EINVAL;
       return 0;
    }

    // Check file name first
    if (allowed_match(pathname, okpath)) return 1;

    // Check directory name
    char* dirpathbuf = strdup(pathname);
    char* dirpath = dirname(dirpathbuf);
    int dir_ok = allowed_match(dirpath, okpath);
    free(dirpathbuf);

    return dir_ok;
}

static int allowed(const char* pathname, int flags) {
    if (flags & O_CREAT)
        return allowed_env(pathname, "ALLOWED_FOR_CREATE");
    else
        return allowed_env(pathname, "ALLOWED_FOR_READ");
}

int open(const char *pathname, int flags, mode_t mode) {
    static int (*real_open)(const char*, int, mode_t) = NULL;
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");

    if (!allowed(pathname, flags)) {
        return -1;
    }

    return real_open(pathname, flags, mode);
}

int creat(const char *pathname, mode_t mode) {
    static int (*real_creat)(const char*, mode_t) = NULL;
    if (!real_creat) real_creat = dlsym(RTLD_NEXT, "creat");

    if (!allowed(pathname, O_CREAT)) {
        return -1;
    }

    return real_creat(pathname, mode);
}

FILE* fopen(const char* name, const char* mode) {
    static FILE* (*real_fopen)(const char*, const char*) = NULL;
    if (!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");

    if (!allowed(name, (mode[0] == 'r') ? 0 : O_CREAT)) {
        return NULL;
    }

    return real_fopen(name, mode);
}
