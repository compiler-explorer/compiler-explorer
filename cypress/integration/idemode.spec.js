function getPossibleFilenames() {
    let d = new Date();

    let possibleFilenames = [];

    for (let s = 0; s < 20; s++) {
        d.setSeconds(d.getSeconds() + 1);

        let datestring = d.getFullYear() +
            ('0' + (d.getMonth() + 1)).slice(-2) +
            ('0' + d.getDate()).slice(-2);
        datestring += ('0' + d.getHours()).slice(-2) +
            ('0' + d.getMinutes()).slice(-2) +
            ('0' + d.getSeconds()).slice(-2);

        possibleFilenames.push('project-' + datestring + '.zip');
    }

    return possibleFilenames;
}

const downloadsFolder = Cypress.config('downloadsFolder');

describe('IDE Mode', () => {
    beforeEach(() => {
        if (Cypress.browser.name !== 'firefox') {
            cy.wrap(
                Cypress.automation('remote:debugger:protocol',
                    {
                        command: 'Page.setDownloadBehavior',
                        params: { behavior: 'allow', downloadPath: downloadsFolder },
                    }),
                { log: false },
            );
        }
    });

    it('Opens IDE Tree pane and Downloads empty project', () => {
        cy.visit('http://127.0.0.1:10240');
        cy.get('#addDropdown').click();
        cy.get('#add-tree').click();
        cy.get('.lm_item .file-menu').click();

        const filenames = getPossibleFilenames();
        cy.get('.lm_item .save-project-to-file').click();

        cy.task('pause', 3000);
        let proms = [];
        for (let filename of filenames) {
            proms.push(new Promise((resolve) => {
                cy.task('readFileMaybe', downloadsFolder + '/' + filename).then(textOrNull => {
                    resolve(!!textOrNull);
                });
            }));
        }

        return Promise.all(proms).then((files) => {
            const filtered = files.filter((val) => val);
            expect(filtered).to.deep.equal([true]);
        });
    });

    it('Upload project in IDE mode', () => {
        cy.visit('http://127.0.0.1:10240');
        cy.get('#addDropdown').click({force: true});
        cy.get('#add-tree').click({force: true});
        cy.get('.lm_item .file-menu').click();

        const filepath = '../resources/project-example.zip';
        cy.get('.lm_item input[type="file"]').attachFile(filepath);

        cy.window().then(window => {
            window.saveGL();
            expect(window.localStorage.gl.includes('CMakeLists.txt')).to.be.true;
        });
    });
});
