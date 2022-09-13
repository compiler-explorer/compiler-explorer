# Sponsorship

Compiler Explorer is funded by sponsors: mostly individuals through Patreon, GitHub sponsors and PayPal one-off
payments. There is potential for corporate sponsorship; though extremely limited due to the goal of making Compiler
Explorer a mostly ad-free experience.

Corporate sponsorship requests should be directed to [Matt](mailto:matt@godbolt.org), who has sole discretion on what
kinds of sponsorship is appropriate.

## The sponsors.yaml format

Sponsors are added by editing the `sponsors.yaml` file. The format is:

```yaml
---
levels:
  - name: Level one
    description: A description of the first level of sponsors.
    class: css-class-to-apply # optional
    sponsors:
      - name: Displayed name
        img: url of an image # optional
        priority: 100 # optional, higher means shown first, else sorted by name
        url: link to navigate to if clicked # optional
        topIcon: true # optional, if true, show on the top of index.pug
      - name: Another...
      - Yet another # top level string treated as a {name: "Yet another"}
  - name: Level two
    description: The second level.
    sponsors:
      - bob
      - ian
      - joe
```
