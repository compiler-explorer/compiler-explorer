const { execSync } = require('child_process');
const { getHashDigest } = require('loader-utils');
const pug = require('pug');
const path = require('path');

// If you edit either cookies.pug or privacy.pug be aware this will trigger a popup on the users' next visit.
// Knowing the last versions here helps us be aware when this happens. If you get an error here and you _haven't_
// knowingly edited either policy, contact the CE team. If you have edited the cookies and know that this expected,
// just update the hash here.
const expectedHashes = {
    cookies: '4bc6a34572c2eb78',
    privacy: 'dcc79570ddaf4bd8',
};

function _execGit(command) {
    const gitResult = execSync(command);
    if (!gitResult) {
        throw new Error(`Failed to execute ${command}`);
    }
    return gitResult.toString();
}

module.exports = function(content) {
    const filePath = this.resourcePath;
    const filename = path.basename(filePath, '.pug');
    const options = this.getOptions();
    if (!options.useGit) {
        this.emitWarning(new Error(`Running without git: file contents for ${filePath} will be wrong`));
    }
    const execGit = options.useGit ? _execGit : () => 'no-git-available';
    const lastTime = execGit(`git log -1 --format=%cd "${filePath}"`).trimEnd();
    const lastCommit = execGit(`git log -1 --format=%h "${filePath}"`).trimEnd();
    const gitChanges = execGit('git log --date=local --after="3 months ago" "--grep=(#[0-9]*)" --oneline')
        .split('\n')
        .map(line => line.match(/(?<hash>\w+) (?<description>.*)/))
        .filter(x => x)
        .map(match => match.groups);

    const compiled = pug.compile(content.toString(), {filename: filePath});

    // When calculating the hash we ignore the hard-to-predict values like lastTime and lastCommit, else every time
    // we merge changes in policies to main we get a new hash after checking in, and that breaks the build.
    const htmlTextForHash = compiled({gitChanges, lastTime:'some-last-time', lastCommit:'some-last-commit'});
    const hashDigest = getHashDigest(htmlTextForHash, 'sha256', 'hex', 16);
    const expectedHash = expectedHashes[filename];
    if (options.useGit && expectedHash !== undefined && expectedHash !== hashDigest) {
        this.emitError(
            new Error(
                `Hash for file '${filePath}' changed from '${expectedHash}' to '${hashDigest}'` +
                    ` - if expected, update the definition in parsed_pug.js`,
            ),
        );
    }

    const htmlText = compiled({gitChanges, lastTime, lastCommit});
    const result = {
        hash: hashDigest,
        text: htmlText,
    };
    return `export default ${JSON.stringify(result)};`;
}
