#!/usr/bin/env node
// Best-effort TextMate (.tmLanguage.json) → Monaco Monarch converter.
//
// Lossy by nature:
//  - Oniguruma features without JS equivalents (atomic groups, possessive
//    quantifiers, \h, \A/\Z) are rewritten to closest JS form.
//  - TextMate scope names are flattened to Monaco token names via prefix map.
//  - begin/end ranges become pushed states; contentName is dropped.
//  - External-grammar includes (`source.foo#bar`) are dropped.
//
// Usage:  node tm-to-monarch.js path/to/grammar.tmLanguage.json > out.js

const fs = require('fs');

const TOKEN_MAP = [
  ['comment.line',                 'comment'],
  ['comment.block.documentation',  'comment.doc'],
  ['comment.block',                'comment'],
  ['comment',                      'comment'],
  ['string.regexp',                'regexp'],
  ['string',                       'string'],
  ['constant.numeric.hex',         'number.hex'],
  ['constant.numeric.octal',       'number.octal'],
  ['constant.numeric.binary',      'number.binary'],
  ['constant.numeric.float',       'number.float'],
  ['constant.numeric',             'number'],
  ['constant.character.escape',    'string.escape'],
  ['constant.character',           'string'],
  ['constant.language',            'keyword'],
  ['constant',                     'constant'],
  ['keyword.operator',             'operator'],
  ['keyword.control',              'keyword'],
  ['keyword',                      'keyword'],
  ['storage.type',                 'type'],
  ['storage.modifier',             'keyword'],
  ['storage',                      'keyword'],
  ['support.function',             'predefined'],
  ['support.type',                 'type'],
  ['support.class',                'type'],
  ['support',                      'type'],
  ['entity.name.function',         'type.identifier'],
  ['entity.name.type',             'type'],
  ['entity.name.tag',              'tag'],
  ['entity.other.attribute-name',  'attribute.name'],
  ['entity',                       'identifier'],
  ['variable.parameter',           'variable.parameter'],
  ['variable.language',            'keyword'],
  ['variable',                     'identifier'],
  ['punctuation.definition.string','string'],
  ['punctuation.definition.comment','comment'],
  ['punctuation',                  'delimiter'],
  ['meta',                         ''],
  ['invalid',                      'invalid'],
];

function mapScope(scope) {
  if (!scope) return '';
  for (const [prefix, token] of TOKEN_MAP) {
    if (scope === prefix || scope.startsWith(prefix + '.')) return token;
  }
  return scope.split('.')[0] || '';
}

function convertRegex(re) {
  if (!re) return re;
  return re
    .replace(/\(\?x\)/g, '')
    .replace(/(\*|\+|\?)\+/g, '$1')
    .replace(/\(\?>/g, '(?:')
    .replace(/\\h/g, '[0-9a-fA-F]')
    .replace(/\\A/g, '^')
    .replace(/\\[Zz]/g, '$');
}

let regexFlags = '';
function makeRegexLiteral(re) {
  return '/' + re.replace(/\//g, '\\/') + '/' + regexFlags;
}

const states = {};
let stateCounter = 0;
const warnings = [];

function processPatterns(patterns, ownerName, ambientToken) {
  const rules = [];
  for (const p of patterns || []) rules.push(...convertPattern(p, ownerName, ambientToken));
  return rules;
}

function convertPattern(p, ownerName, ambientToken) {
  if (p.include) {
    let name = p.include;
    if (name.startsWith('#')) name = name.slice(1);
    else if (name === '$self' || name === '$base') name = 'root';
    else { warnings.push(`dropped external include: ${p.include}`); return []; }
    return [{ include: '@' + name }];
  }
  if (p.match) {
    try { new RegExp(convertRegex(p.match)); }
    catch (e) { warnings.push(`bad regex (${e.message}): ${p.match}`); return []; }
    return [['rule', convertRegex(p.match), buildAction(p)]];
  }
  if (p.begin && p.end) {
    const stateName = (ownerName || 'state') + '_' + (++stateCounter);
    try { new RegExp(convertRegex(p.begin)); new RegExp(convertRegex(p.end)); }
    catch (e) { warnings.push(`bad begin/end regex (${e.message}): ${p.begin} / ${p.end}`); return []; }

    const beginAction = buildRangeAction(p.beginCaptures, p.name, '@' + stateName);
    const endAction   = buildRangeAction(p.endCaptures,   p.name, '@pop');

    // In TextMate, a begin/end range with `name` (or `contentName`) gives that
    // scope to everything between the markers. Inherit it as the ambient token
    // for nested states that don't specify their own.
    const ownToken = mapScope(p.contentName || p.name);
    const inherited = ownToken || ambientToken || '';

    const innerRules = processPatterns(p.patterns, stateName, inherited);
    innerRules.push(['rule', convertRegex(p.end), endAction]);
    if (inherited) innerRules.push(['rule', '.', inherited]);
    states[stateName] = innerRules;

    return [['rule', convertRegex(p.begin), beginAction]];
  }
  if (p.patterns) return processPatterns(p.patterns, ownerName, ambientToken);
  return [];
}

function buildAction(p) {
  const baseToken = mapScope(p.name);
  if (!p.captures) return baseToken || '';
  const keys = Object.keys(p.captures).map(Number).filter(n => !isNaN(n));
  if (keys.length === 0) return baseToken || '';
  if (keys.length === 1 && keys[0] === 0) {
    return mapScope(p.captures['0'].name) || baseToken || '';
  }
  const max = Math.max(...keys);
  const arr = [];
  for (let i = 1; i <= max; i++) {
    const c = p.captures[String(i)];
    arr.push((c && mapScope(c.name)) || '');
  }
  return arr;
}

function buildRangeAction(captures, fallbackName, next) {
  const baseToken = mapScope(fallbackName);
  if (!captures) {
    return next === '@pop' && !baseToken
      ? { token: '', next }
      : { token: baseToken || '', next };
  }
  const keys = Object.keys(captures).map(Number).filter(n => !isNaN(n));
  if (keys.length === 0) return { token: baseToken || '', next };
  if (keys.length === 1 && keys[0] === 0) {
    return { token: mapScope(captures['0'].name) || baseToken || '', next };
  }
  const max = Math.max(...keys);
  const arr = [];
  for (let i = 1; i <= max; i++) {
    const c = captures[String(i)];
    arr.push((c && mapScope(c.name)) || baseToken || '');
  }
  const last = arr[arr.length - 1];
  arr[arr.length - 1] = { token: last, next };
  return arr;
}

// Walk include chains to find every push (next: '@X') reachable from a state.
function expandedRules(name, seen = new Set()) {
  if (seen.has(name)) return [];
  seen.add(name);
  const out = [];
  for (const r of states[name] || []) {
    if (r.include) {
      out.push(...expandedRules(r.include.replace(/^@/, ''), seen));
    } else {
      out.push(r);
    }
  }
  return out;
}

function ambientOf(name) {
  for (const r of states[name] || []) {
    if (Array.isArray(r) && r[1] === '.' && typeof r[2] === 'string' && r[2]) return r[2];
  }
  return null;
}

// True if state `name` is the target of any push (`next: '@name'`).
// Include-only states aren't push targets — adding a catch-all to one would
// shadow the includer's own end/pop rule.
function isPushTarget(name) {
  for (const other of Object.keys(states)) {
    for (const r of states[other] || []) {
      if (Array.isArray(r) && r[2] && typeof r[2] === 'object' && r[2].next === '@' + name) return true;
    }
  }
  return false;
}

// Walk caller chains (push or include) to gather all ambient tokens that the
// parser could be carrying when execution reaches `name`. Passthrough states
// (no own ambient) are followed transitively; cycles are guarded.
function effectiveAmbients(name, visited = new Set()) {
  if (visited.has(name)) return new Set();
  visited.add(name);
  const own = ambientOf(name);
  if (own) return new Set([own]);
  const result = new Set();
  for (const other of Object.keys(states)) {
    if (other === name) continue;
    let isCaller = false;
    for (const r of expandedRules(other)) {
      if (Array.isArray(r) && r[2] && typeof r[2] === 'object' && r[2].next === '@' + name) { isCaller = true; break; }
    }
    if (!isCaller) {
      for (const r of states[other] || []) {
        if (r.include === '@' + name) { isCaller = true; break; }
      }
    }
    if (isCaller) for (const a of effectiveAmbients(other, visited)) result.add(a);
  }
  return result;
}

function propagateAmbient() {
  for (const name of Object.keys(states)) {
    if (name === 'root') continue;
    if (ambientOf(name)) continue;
    if (!isPushTarget(name)) continue;
    const candidates = effectiveAmbients(name);
    if (candidates.size === 1) {
      const [t] = candidates;
      if (t) states[name].push(['rule', '.', t]);
    }
  }
}

function printAction(a) {
  if (typeof a === 'string') return JSON.stringify(a);
  if (Array.isArray(a)) return '[' + a.map(printAction).join(', ') + ']';
  const parts = Object.entries(a).map(([k, v]) => `${k}: ${JSON.stringify(v)}`);
  return '{ ' + parts.join(', ') + ' }';
}

function printRule(rule, indent) {
  if (rule.include) return `${indent}{ include: ${JSON.stringify(rule.include)} },`;
  const [, re, action] = rule;
  return `${indent}[${makeRegexLiteral(re)}, ${printAction(action)}],`;
}

function main() {
  const argv = process.argv.slice(2);
  const positional = [];
  for (const a of argv) {
    if (a === '-i' || a === '--case-insensitive') regexFlags = 'i';
    else positional.push(a);
  }
  const path = positional[0];
  const langId = positional[1];
  if (!path) {
    console.error('usage: tm-to-monarch.js [-i|--case-insensitive] grammar.tmLanguage.json [language-id]');
    process.exit(1);
  }
  const grammar = JSON.parse(fs.readFileSync(path, 'utf8'));
  const scopeTail = (grammar.scopeName || 'lang').split('.').slice(-1)[0];
  const id = langId || scopeTail;
  const postfix = '.' + scopeTail;

  states.root = processPatterns(grammar.patterns, 'root');
  for (const [name, def] of Object.entries(grammar.repository || {})) {
    states[name] = processPatterns(def.patterns || [def], name);
  }
  propagateAmbient();

  const out = [];
  out.push(`// Generated from ${path} by tm-to-monarch.js`);
  out.push(`// Best-effort conversion — review before use.`);
  if (warnings.length) {
    out.push(`// Warnings:`);
    for (const w of warnings) out.push(`//   - ${w}`);
  }
  out.push('');
  out.push(`import * as monaco from 'monaco-editor';`);
  out.push('');
  out.push(`export function definition(): monaco.languages.IMonarchLanguage {`);
  out.push(`    return {`);
  out.push(`        defaultToken: '',`);
  out.push(`        tokenPostfix: ${JSON.stringify(postfix)},`);
  out.push(`        tokenizer: {`);
  for (const [name, rules] of Object.entries(states)) {
    out.push(`            ${name}: [`);
    for (const r of rules) out.push(printRule(r, '                '));
    out.push(`            ],`);
  }
  out.push(`        },`);
  out.push(`    };`);
  out.push(`}`);
  out.push(`monaco.languages.register({id: ${JSON.stringify(id)}});`);
  out.push(`monaco.languages.setMonarchTokensProvider(${JSON.stringify(id)}, definition());`);
  out.push('');
  process.stdout.write(out.join('\n'));
  if (warnings.length) {
    process.stderr.write(`\n${warnings.length} warning(s):\n`);
    for (const w of warnings) process.stderr.write(`  - ${w}\n`);
  }
}

main();
