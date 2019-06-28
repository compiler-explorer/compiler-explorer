"use strict";

const
    Demangler = require("./demangler").Demangler,
    utils = require('./utils');

class DemanglerRustDirect extends Demangler {
    getInput() {
        this.input = this.result.asm.map(line => line.text);
        return this.input.join("\n");
    }

    processOutput(output) {
        const lines = utils.splitLines(output.stdout);
        for (let i = 0; i < this.result.asm.length; ++i) {
            this.result.asm[i].text = lines[i];
        }

        return this.result;
    }
}

exports.Demangler = DemanglerRustDirect;
