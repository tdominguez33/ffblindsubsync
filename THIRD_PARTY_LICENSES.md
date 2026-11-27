# Third-Party Licenses

`ffblindsubsync` is licensed under the MIT License (see [LICENSE](LICENSE)).
It includes or links against the following third-party open-source and freeware components for archive extraction and media processing:

## libarchive (2-Clause BSD License)

Used via `libarchive.js` (WebAssembly) in the browser build and optionally via system `bsdtar`/`tar` for reading `.zip`, `.rar`, `.7z`, and `.tar.*` archives.

```text
Copyright (c) 2003-2024 Tim Kientzle and libarchive contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer
   in this position and unchanged.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## libarchive.js (MIT License)

Copyright (c) 2018 ნიკა — https://github.com/nika-begiashvili/libarchivejs  
Licensed under the MIT License.

## py7zr (GNU Lesser General Public License v2.1 or later)

Used in CLI and Windows Desktop builds for `.7z` archive extraction.  
Copyright (c) 2019-2025 Hiroshi Miura and contributors — https://github.com/miurahr/py7zr  
Licensed under the LGPL-2.1-or-later.

## unrar2-cffi (Apache License 2.0) & RARLAB UnRAR Library

Used in CLI and Windows Desktop builds for `.rar` (RAR4/RAR5) decompression only.  
- `unrar2-cffi` Python bindings: Apache-2.0 — https://github.com/v4/unrar2-cffi
- RARLAB UnRAR decompression library: Freeware UnRAR license (decompression permitted for free redistribution; creating RAR archives is prohibited and not performed by this software).
