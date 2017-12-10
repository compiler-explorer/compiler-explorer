// Copyright (c) 2012-2017, Rubén Rincón
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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const should = require('chai').should();
const properties = require('../lib/properties');


properties.initialize('test/example-config/', ['test', 'overridden-base', 'overridden-tip']);

const defaultProps = properties.propsFor("default");
const overridingProps = properties.propsFor("overwrite");

describe('Properties', () => {
    it('Has working propsFor', () => {
        should.equal(properties.get("default", "exampleProperty"), defaultProps("exampleProperty"));
    });
    it('Does not find non existent properties when no default is set', () => {
        should.equal(defaultProps("nonexistentProp"), undefined);
    });
    it('Falls back to default if value not found and default is set', () => {
        // Randomly generated number...
        defaultProps("nonexistentProp", 4).should.be.equal(4);
        should.equal(defaultProps("nonexistentProp", 4), 4);
    });
    it('Handles empty properties as empty strings', ()  => {
        should.equal(defaultProps("emptyProperty"), "");
    });
    it('Ignores commented out properties', () => {
        should.equal(defaultProps("commentedProperty"), undefined);
    });
    it('Understands positive integers', () => {
        should.equal(defaultProps("numericPropertyPositive"), 42);
    });
    it('Understands zero as integer', () => {
        should.equal(defaultProps("numericPropertyZero"), 0);
    });
    it('Understands negative integers', () => {
        should.equal(defaultProps("numericPropertyNegative"), -11);
    });
    it('Understands positive floats', () => {
        should.equal(defaultProps("floatPropertyPositive"), 3.14);
    });
    it('Understands negative floats', () => {
        should.equal(defaultProps("floatPropertyNegative"), -9000.0);
    });
    it('Does not understand comma decimal as float', () => {
        should.equal(defaultProps("commaAsDecimalProperty"), "3,14");
    });
    it('Does not understand DASH-SPACE-NUMBER as a negative number', () => {
        should.equal(defaultProps("stringPropertyNumberLike"), "- 97");
    });
    it('Understands yes as true boolean', () => {
        should.equal(defaultProps("truePropertyYes"), true);
    });
    it('Understands true as true boolean', () => {
        should.equal(defaultProps("truePropertyTrue"), true);
    });
    it('Does not understand Yes as boolean', () => {
        should.equal(defaultProps("stringPropertyYes"), "Yes");
    });
    it('Does not understand True as boolean', () => {
        should.equal(defaultProps("stringPropertyTrue"), "True");
    });
    it('Understands no as false boolean', () => {
        should.equal(defaultProps("falsePropertyNo"), false);
    });
    it('Understands false as false boolean', () => {
        should.equal(defaultProps("falsePropertyFalse"), false);
    });
    it('Does not understand No as boolean', () => {
        should.equal(defaultProps("stringPropertyNo"), "No");
    });
    it('Does not understand False as boolean', () => {
        should.equal(defaultProps("stringPropertyFalse"), "False");
    });
    it('Should find non overridden properties', () => {
        should.equal(overridingProps("nonOverriddenProperty"), ".... . .-.. .-.. ---");
    });
    it('Should handle overridden properties', () => {
        should.equal(overridingProps("overrodeProperty"), "ACTUALLY USED");
    });
    it('Should fall back from overridden', () => {
        should.equal(overridingProps("localProperty"), 11235813);
    });
});