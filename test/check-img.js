import fs from 'fs';
import path from 'path';

import {assert} from 'chai';

import {languages} from '../lib/languages';

const img_dir = path.resolve('views/resources/logos');

function checkImage(lang) {
    let result = fs.existsSync(img_dir + '/' + lang.logoUrl);
    if (lang.logoUrlDark !== null) {
        result = result && fs.existsSync(img_dir + '/' + lang.logoUrlDark);
    }
    return result;
}

describe('Image-checks', () => {
    for (const lang in languages) {
        it('check if ' + lang + ' image exists', () => assert.isOk(checkImage(languages[lang])));
    }
});
