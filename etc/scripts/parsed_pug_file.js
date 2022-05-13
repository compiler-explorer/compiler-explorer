import {execSync} from 'child_process';
import {getHashDigest} from 'loader-utils';
import * as pug from 'pug';
import path from 'path';

// If you edit either cookies.pug or privacy.pug be aware this will trigger a popup on the users' next visit.
// Knowing the last versions here helps us be aware when this happens. If you get an error here and you _haven't_
// knowingly edited either policy, contact the CE team. If you have edited the cookies and know that this expected,
// just update the hash here. Note that you will need to check in the changes to the policy document first, then
// run `make webpack`, and then paste in those values, as the timestamp of the checkin is part of the hash.
const expectedHashes = {
    cookies: '7581921381afeed5',
    privacy: 'a0540b2506625656',
};

function execGit(command) {
    const gitResult = execSync(command);
    if (!gitResult) {
        throw new Error(`Failed to execute ${command}`);
    }
    return gitResult.toString();
}

const gitChanges = execGit('git log --date=local --after="3 months ago" "--grep=(#[0-9]*)" --oneline')
    .split('\n')
    .map(line => line.match(/(?<hash>\w+) (?<description>.*)/))
    .filter(x => x)
    .map(match => match.groups);

export default function(content) {
    const filePath = this.resourcePath;
    const filename = path.basename(filePath, '.pug');
    const lastTime = execGit(`git log -1 --format=%cd "${filePath}"`).trimEnd();
    const lastCommit = execGit(`git log -1 --format=%h "${filePath}"`).trimEnd();
    const compiled = pug.compile(content.toString(), {filename: filePath});
    const source = compiled({gitChanges, lastTime, lastCommit});
    const hashDigest = getHashDigest(source, 'sha256', 'hex', 16);
    const expectedHash = expectedHashes[filename];
    if (expectedHash !== undefined && expectedHash !== hashDigest) {
        this.emitError(
            new Error(
                `Hash for file '${filePath}' changed from '${expectedHash}' to '${hashDigest}'` +
                    ` - if expected, update the definition in parsed_pug.js`,
            ),
        );
    }
    const result = {
        hash: hashDigest,
        text: source.toString(),
    };
    return `export default ${JSON.stringify(result)};`;
}
