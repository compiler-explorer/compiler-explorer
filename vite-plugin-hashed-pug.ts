import {execSync} from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import {getHashDigest} from 'loader-utils';
import * as pug from 'pug';
import {Plugin, createLogger} from 'vite';

const HASHES = {
    cookies: '08712179739d3679',
    privacy: 'c0dad1f48a56b761',
} as const;

export type VitePluginHashedPugOptions = {
    useGit?: boolean;
};

/**
 * Vite plugin to handle our logic for hashed Pug templates to be built into the static assets.
 *
 * Some of our documents (privacy policy, cookie policy) are sensitive in the sense the user should
 * be informed when they are changed. For this reason, we keep a hash of the pug template and
 * notify the user when it has changed.
 */
export function vitePluginHashedPug({useGit}): Plugin {
    const logger = createLogger('info', {
        prefix: 'vite-plugin-hashed-pug',
    });
    if (!useGit) {
        logger.warn('vite-plugin-hashed-pug is configured to not use git: file contents will be wrong');
    }
    const execGit = useGit ? execGitSync : () => 'no-git-available';
    return {
        name: 'vite-plugin-hashed-pug',
        async transform(src, id) {
            if (!id.endsWith('.pug')) {
                return null;
            }
            const filename = path.basename(id);
            const lastTime = execGit(`git log -1 --format=%cd "${id}"`).trimEnd();
            const lastCommit = execGit(`git log -1 --format=%h "${id}"`).trimEnd();
            const gitChanges = execGit('git log --date=local --after="3 months ago" "--grep=(#[0-9]*)" --oneline')
                .split('\n')
                .map(line => line.match(/(?<hash>\w+) (?<description>.*)/))
                .filter(x => x)
                .map(match => match?.groups ?? {});
            const content = await fs.readFile(id, 'utf-8');
            const pugModule = pug.compile(content, {filename: id});
            // When calculating the hash we ignore the hard-to-predict values like lastTime and lastCommit, else every
            // time we merge changes in policies to main we get a new hash after checking in, and that breaks the build.
            const htmlTextForHash = pugModule({gitChanges, lastTime: 'some-last-time', lastCommit: 'some-last-commit'});
            const hashDigest = getHashDigest(htmlTextForHash, 'sha256', 'hex', 16);
            const expectedHash = HASHES[filename];
            if (useGit && expectedHash !== undefined && expectedHash !== hashDigest) {
                logger.warn(`Hash mismatch for ${filename}: expected ${expectedHash}, got ${hashDigest}`);
                logger.warn('If this was expected, update the hash in `vite-plugin-hashed-pug.ts`');
            }

            const html = pugModule({gitChanges, lastTime, lastCommit});
            return {
                code: `
                export default {
                    hash: '${hashDigest}',
                    text: \`${html}\`,
                };
                `.trim(),
            };
        },
    };
}

function execGitSync(command: string) {
    const result = execSync(command);
    const stdout = result.toString().trim();
    if (stdout === '') {
        throw new Error(`Failed to execute ${command} (empty stdout)`);
    }
    return stdout;
}
