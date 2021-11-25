describe('Basic rendering', () => {
  it('Has correct webpage title', () => {
    cy.visit('http://127.0.0.1:10240');
    cy.get('title').should('have.html', 'Compiler Explorer');
  });
});
