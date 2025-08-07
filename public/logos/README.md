# Logo Optimization Guidelines

This directory contains logos for programming languages displayed in Compiler Explorer. To maintain fast loading times and reasonable bundle sizes, logos should be optimized before adding them.

## Size Guidelines

- **Target size**: Keep individual logos under 20KB when possible
- **Maximum dimensions**: 256×256 pixels (preserving aspect ratio)
- **File formats**: SVG preferred for simple graphics, PNG for complex images

## Optimization Tools & Commands

### PNG Images
```bash
# Resize and optimize (preserves aspect ratio)
convert input.png -resize 256x256> -quality 85 output.png

# Alternative with pngquant for better compression
pngquant --quality=65-80 --output output.png input.png
```

### SVG Images
```bash
# Optimize with SVGO (install with: npm install -g svgo)
npx svgo input.svg --output output.svg

# For SVGs with embedded raster data, convert to PNG instead
convert input.svg -resize 256x256 -quality 85 output.png
```

## Common Issues

- **Embedded Base64 data in SVGs**: These are usually very large (300KB+). Convert to PNG instead.
- **Oversized PNGs**: Images over 1000px wide/tall should be resized to 256×256 or smaller.
- **16-bit PNGs**: Convert to 8-bit with `convert input.png -depth 8 output.png`

## Recent Optimizations

Examples of successful optimizations:
- `hylo.svg` (321KB) → `hylo.png` (27KB) - 92% reduction
- `nim.svg` (126KB) → `nim.png` (13KB) - 90% reduction  
- `scala.png` (79KB) → optimized (14KB) - 82% reduction

Always test that optimized logos look good at small icon sizes (16-32px) in the UI.