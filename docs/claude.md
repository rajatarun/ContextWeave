# PlantUML Guidelines

## Asset Management

### Always download assets locally

Do **not** reference remote URLs for assets (icons, sprites, stdlib includes) in PlantUML files.
Download all required assets locally and reference them via relative paths.

**Wrong:**
```plantuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml
!include https://raw.githubusercontent.com/plantuml-stdlib/AWS-PlantUML/master/awslib/AWSCommon.puml
```

**Correct:**
```plantuml
!include ../assets/plantuml/C4_Context.puml
!include ../assets/plantuml/awslib/AWSCommon.puml
```

Place downloaded assets under `assets/plantuml/` so they are versioned with the repository and available offline.

## PNG Export

Always generate a PNG file after creating or modifying a `.puml` file.

```bash
# Using the PlantUML jar
java -jar plantuml.jar docs/my-diagram.puml

# Using the plantuml CLI (if installed)
plantuml docs/my-diagram.puml
```

The resulting `.png` must be committed alongside the `.puml` source so diagrams are immediately viewable on GitHub without requiring a local PlantUML installation.

Naming convention: the PNG must share the same base name as the PUML file.

| Source | Generated PNG |
|--------|--------------|
| `docs/c4.puml` | `docs/c4.png` |
| `docs/aws-infrastructure.puml` | `docs/aws-infrastructure.png` |
