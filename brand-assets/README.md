# Odin Brand Assets

Vector brand assets for **Odin**, traced from the master compass mark and
built into the same system used across Prescott Data projects.

## Files

`svg/` contains three lockups, each in five variants:

| Type | Description |
|------|-------------|
| `monogram-*` | The compass mark on its own (favicon, app icon, avatar) |
| `wordmark-*` | The "Odin" logotype on its own |
| `combo-*` | Mark + logotype lockup (primary logo) |

| Variant | Use |
|---------|-----|
| `*-brand` | Brand blue `#1758F5` on light backgrounds |
| `*-black` | Ink navy `#0B1220` for monochrome light backgrounds |
| `*-white` | White `#FFFFFF` for dark backgrounds |
| `*-purple` | Accent purple `#7D55FA` |
| `*-trimmed` | Ink navy, tightly cropped bounding box (for precise layout) |

## Palette

| Token | Hex |
|-------|-----|
| Brand | `#1758F5` |
| Ink | `#0B1220` |
| Purple | `#7D55FA` |
| White | `#FFFFFF` |

## Notes

- The mark is a single traced path (potrace), so every asset is resolution
  independent and renders crisply at any size.
- The wordmark is set in Avenir Next Demi Bold, outlined to paths.
- To recolor, edit the `fill` attribute on the `<g>` element.
