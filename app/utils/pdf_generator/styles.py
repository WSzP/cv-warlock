"""CV PDF style configuration.

Defines visual styles (plain, modern) with their color palettes,
typography settings, layout parameters, and feature flags.
"""

from dataclasses import dataclass
from enum import StrEnum


class CVStyle(StrEnum):
    """Available CV PDF styles."""

    PLAIN = "plain"
    MODERN = "modern"


@dataclass
class StyleConfig:
    """Configuration for a CV style."""

    # Colors (RGB tuples)
    accent_color: tuple[int, int, int]
    accent_dark: tuple[int, int, int]  # Darker shade for gradients/hover
    accent_light: tuple[int, int, int]  # Light tint for backgrounds
    text_primary: tuple[int, int, int]
    text_secondary: tuple[int, int, int]
    text_muted: tuple[int, int, int]
    text_on_accent: tuple[int, int, int]  # Text color on accent backgrounds
    divider_color: tuple[int, int, int]
    card_background: tuple[int, int, int]  # Subtle background for cards
    link_color: tuple[int, int, int]  # Clickable link color (distinct from accent)

    # Typography
    name_size: int
    section_header_size: int
    job_title_size: int
    body_size: int
    contact_size: int

    # Layout
    left_margin: float
    top_margin: float
    right_margin: float
    section_spacing: float
    entry_spacing: float

    # Style features
    use_header_band: bool  # Full-width colored header
    header_band_height: float
    use_accent_bar: bool  # Left side accent bar
    accent_bar_width: float
    section_header_uppercase: bool
    section_header_underline: bool
    section_header_band: bool  # Full colored background for section headers
    use_skill_pills: bool  # Display skills as pill badges
    use_entry_cards: bool  # Subtle background for job entries
    contact_separator: str
    inline_logo_height: float = 3.5  # Height (mm) of company logos in job titles


# Style presets
STYLE_CONFIGS: dict[CVStyle, StyleConfig] = {
    CVStyle.PLAIN: StyleConfig(
        # Colors - all grayscale for plain
        accent_color=(0, 0, 0),
        accent_dark=(0, 0, 0),
        accent_light=(240, 240, 240),
        text_primary=(0, 0, 0),
        text_secondary=(64, 64, 64),
        text_muted=(96, 96, 96),
        text_on_accent=(255, 255, 255),
        divider_color=(200, 200, 200),
        card_background=(250, 250, 250),
        link_color=(0, 68, 204),  # #0044CC - distinct clickable link color
        # Typography
        name_size=18,
        section_header_size=16,
        job_title_size=12,
        body_size=10,
        contact_size=10,
        # Layout
        left_margin=20.0,
        top_margin=20.0,
        right_margin=20.0,
        section_spacing=6.0,
        entry_spacing=6.0,
        # Style features
        use_header_band=False,
        header_band_height=0.0,
        use_accent_bar=False,
        accent_bar_width=0.0,
        section_header_uppercase=True,
        section_header_underline=True,
        section_header_band=False,
        use_skill_pills=False,
        use_entry_cards=False,
        contact_separator=" | ",
    ),
    CVStyle.MODERN: StyleConfig(
        # Colors - sophisticated deep navy palette
        accent_color=(20, 50, 90),  # Deep navy blue
        accent_dark=(15, 35, 65),  # Darker navy for depth
        accent_light=(235, 242, 250),  # Very light blue tint
        text_primary=(25, 30, 38),  # Near-black with warmth
        text_secondary=(55, 65, 80),  # Dark slate
        text_muted=(90, 100, 115),  # Medium slate
        text_on_accent=(255, 255, 255),  # White on accent
        divider_color=(220, 228, 238),  # Subtle blue-gray divider
        card_background=(247, 250, 253),  # Very subtle blue tint
        link_color=(0, 68, 204),  # #0044CC - distinct clickable link color
        # Typography - larger, bolder
        name_size=26,
        section_header_size=11,
        job_title_size=11,
        body_size=10,
        contact_size=9,
        # Layout - generous spacing
        left_margin=20.0,
        top_margin=15.0,
        right_margin=20.0,
        section_spacing=10.0,
        entry_spacing=8.0,
        # Style features
        use_header_band=True,
        header_band_height=38.0,
        use_accent_bar=False,
        accent_bar_width=0.0,
        section_header_uppercase=True,
        section_header_underline=False,
        section_header_band=True,
        use_skill_pills=True,
        use_entry_cards=False,
        contact_separator="  \u2022  ",
    ),
}
