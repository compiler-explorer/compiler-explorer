// Copyright (c) 2022, Compiler Explorer Authors
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

//import {stub} from 'sinon';
//import {isValidAd} from '../static/motd';

// describe('Motd', () => {
//     it("Doesn't filter ads if no lang is set", () => {
//         const ad = {
//             filter: [],
//             html: '',
//         };
//         isValidAd(ad, 'langForTest').should.be.true;
//     });
//     it('Filters ad if not the correct language', () => {
//         const ad = {
//             filter: ['anotherLang'],
//             html: '',
//         };
//         isValidAd(ad, 'langForTest').should.be.false;
//     });
//     it("Doesn't filter ad if the correct language is used", () => {
//         const ad = {
//             filter: ['langForTest'],
//             html: '',
//         };
//         isValidAd(ad, 'langForTest').should.be.true;
//     });
//     it("Doesn't filter ad if now > from", () => {
//         const dateNowStub = stub(Date, 'now');
//         dateNowStub.returns(1641596400000); // 2022-01-08T00:00:00
//         const ad = {
//             filter: [],
//             html: '',
//             valid_from: '2022-01-01T00:00:00',
//         };
//         isValidAd(ad, 'langForTest').should.be.true;
//     });
//     it('Filter ad if now < from', () => {
//         const dateNowStub = stub(Date, 'now');
//         dateNowStub.returns(1641596400000); // 2022-01-08T00:00:00
//         const ad = {
//             filter: [],
//             html: '',
//             valid_from: '2022-01-016T00:00:00',
//         };
//         isValidAd(ad, 'langForTest').should.be.false;
//     });
//     it("Doesn't filter ad if now < until", () => {
//         const dateNowStub = stub(Date, 'now');
//         dateNowStub.returns(1641596400000); // 2022-01-08T00:00:00
//         const ad = {
//             filter: [],
//             html: '',
//             valid_from: '2022-01-016T00:00:00',
//         };
//         isValidAd(ad, 'langForTest').should.be.true;
//     });
//     it('Filter ad if now > until', () => {
//         const dateNowStub = stub(Date, 'now');
//         dateNowStub.returns(1641596400000); // 2022-01-08T00:00:00
//         const ad = {
//             filter: [],
//             html: '',
//             valid_from: '2022-01-016T00:00:00',
//         };
//         isValidAd(ad, 'langForTest').should.be.false;
//     });
// });
