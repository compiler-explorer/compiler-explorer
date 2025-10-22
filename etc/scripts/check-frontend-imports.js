#!/usr/bin/env node
// Check that frontend (static/) code doesn't import from backend (lib/)

import {execSync} from 'child_process';

try {
    const violations = execSync(
        'git grep -n "from [\'\\\"]\\.\\./.*lib/" -- "static/*.ts" "static/**/*.ts"',
        {encoding: 'utf8'}
    ).trim();

    if (violations) {
        console.error('❌ Error: Frontend code cannot import from backend (lib/) directory!');
        console.error('');
        console.error('Found violations:');
        console.error(violations);
        console.error('');
        console.error('Frontend code should use API calls or import from types/ instead of directly importing backend code.');
        process.exit(1);
    }
} catch (e) {
    // git grep returns non-zero when no matches found, which is what we want
    if (e.status !== 1) {
        console.error('Error running git grep:', e.message);
        process.exit(1);
    }
}

process.exit(0);
