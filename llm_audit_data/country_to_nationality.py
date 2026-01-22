"""
Country to Nationality Converter

This module converts country names to their corresponding nationality adjectives
(e.g., "Finland" → "Finnish", "United States" → "American"). It handles many
country name variations and special cases to ensure accurate conversion for use
in persona descriptions.

Usage:
    from country_to_nationality import country_to_nationality
    nationality = country_to_nationality("Finland")  # Returns "Finnish"
"""

COUNTRY_TO_NATIONALITY = {
    # Major countries
    'United States': 'American',
    'United States of America': 'American',
    'USA': 'American',
    'Canada': 'Canadian',
    'Mexico': 'Mexican',
    'Brazil': 'Brazilian',
    'Argentina': 'Argentine',
    'Chile': 'Chilean',
    'Colombia': 'Colombian',
    'Peru': 'Peruvian',
    'Venezuela': 'Venezuelan',
    'Ecuador': 'Ecuadorian',
    'Uruguay': 'Uruguayan',
    'Paraguay': 'Paraguayan',
    'Bolivia': 'Bolivian',
    'Costa Rica': 'Costa Rican',
    'Panama': 'Panamanian',
    'Guatemala': 'Guatemalan',
    'El Salvador': 'Salvadoran',
    'Honduras': 'Honduran',
    'Nicaragua': 'Nicaraguan',
    
    # European countries
    'United Kingdom': 'British',
    'UK': 'British',
    'Germany': 'German',
    'France': 'French',
    'Italy': 'Italian',
    'Spain': 'Spanish',
    'Portugal': 'Portuguese',
    'Greece': 'Greek',
    'Netherlands': 'Dutch',
    'Belgium': 'Belgian',
    'Switzerland': 'Swiss',
    'Austria': 'Austrian',
    'Sweden': 'Swedish',
    'Norway': 'Norwegian',
    'Denmark': 'Danish',
    'Finland': 'Finnish',
    'Iceland': 'Icelandic',
    'Ireland': 'Irish',
    'Poland': 'Polish',
    'Czechia': 'Czech',
    'Czech Republic': 'Czech',
    'Slovakia': 'Slovak',
    'Slovenia': 'Slovenian',
    'Croatia': 'Croatian',
    'Serbia': 'Serbian',
    'Romania': 'Romanian',
    'Bulgaria': 'Bulgarian',
    'Hungary': 'Hungarian',
    'Lithuania': 'Lithuanian',
    'Latvia': 'Latvian',
    'Estonia': 'Estonian',
    'Ukraine': 'Ukrainian',
    'Russia': 'Russian',
    'Belarus': 'Belarusian',
    'Luxembourg': 'Luxembourgish',
    'Malta': 'Maltese',
    'Cyprus': 'Cypriot',
    
    # Asian countries
    'China': 'Chinese',
    'Japan': 'Japanese',
    'South Korea': 'South Korean',
    'Korea': 'Korean',
    'India': 'Indian',
    'Pakistan': 'Pakistani',
    'Bangladesh': 'Bangladeshi',
    'Thailand': 'Thai',
    'Vietnam': 'Vietnamese',
    'Philippines': 'Filipino',
    'Indonesia': 'Indonesian',
    'Malaysia': 'Malaysian',
    'Singapore': 'Singaporean',
    'Taiwan': 'Taiwanese',
    'Hong Kong': 'Hong Kong',
    'Mongolia': 'Mongolian',
    'Nepal': 'Nepalese',
    'Sri Lanka': 'Sri Lankan',
    'Afghanistan': 'Afghan',
    'Bhutan': 'Bhutanese',
    'Myanmar': 'Myanmar',
    'Cambodia': 'Cambodian',
    'Laos': 'Laotian',
    
    # Middle East
    'Israel': 'Israeli',
    'Saudi Arabia': 'Saudi',
    'United Arab Emirates': 'Emirati',
    'UAE': 'Emirati',
    'Qatar': 'Qatari',
    'Kuwait': 'Kuwaiti',
    'Bahrain': 'Bahraini',
    'Oman': 'Omani',
    'Jordan': 'Jordanian',
    'Lebanon': 'Lebanese',
    'Turkey': 'Turkish',
    'Iran': 'Iranian',
    'Iraq': 'Iraqi',
    'Yemen': 'Yemeni',
    'Syria': 'Syrian',
    'Palestine': 'Palestinian',
    
    # African countries
    'South Africa': 'South African',
    'Egypt': 'Egyptian',
    'Nigeria': 'Nigerian',
    'Kenya': 'Kenyan',
    'Ghana': 'Ghanaian',
    'Senegal': 'Senegalese',
    'Ethiopia': 'Ethiopian',
    'Morocco': 'Moroccan',
    'Algeria': 'Algerian',
    'Tunisia': 'Tunisian',
    'Libya': 'Libyan',
    'Tanzania': 'Tanzanian',
    'Uganda': 'Ugandan',
    'Zimbabwe': 'Zimbabwean',
    'Mozambique': 'Mozambican',
    'Madagascar': 'Malagasy',
    'Mauritius': 'Mauritian',
    'Rwanda': 'Rwandan',
    'Cameroon': 'Cameroonian',
    'Ivory Coast': 'Ivorian',
    'Sudan': 'Sudanese',
    'Angola': 'Angolan',
    
    # Oceania
    'Australia': 'Australian',
    'New Zealand': 'New Zealander',
    'Fiji': 'Fijian',
    'Papua New Guinea': 'Papua New Guinean',
    
    # Other
    'Haiti': 'Haitian',
    'Jamaica': 'Jamaican',
    'Trinidad and Tobago': 'Trinidadian',
    'Barbados': 'Barbadian',
    'Dominican Republic': 'Dominican',
    'Cuba': 'Cuban',
    'Kazakhstan': 'Kazakh',
    'Uzbekistan': 'Uzbek',
    'Azerbaijan': 'Azerbaijani',
    'Armenia': 'Armenian',
    'Georgia': 'Georgian',
    
    # Additional countries from dataset
    'Albania': 'Albanian',
    'Belize': 'Belizean',
    'Benin': 'Beninese',
    'Botswana': 'Motswana',
    'Bosnia and Herzegovina': 'Bosnian',
    'Burkina Faso': 'Burkinabe',
    'Burundi': 'Burundian',
    'Central African Republic': 'Central African',
    'Chad': 'Chadian',
    'Comoros': 'Comoran',
    'Congo': 'Congolese',
    'DR Congo': 'Congolese',
    'Democratic Republic of the Congo': 'Congolese',
    'Côte d\'Ivoire': 'Ivorian',
    'Cote d\'Ivoire': 'Ivorian',
    'Djibouti': 'Djiboutian',
    'Equatorial Guinea': 'Equatorial Guinean',
    'Eritrea': 'Eritrean',
    'Eswatini': 'Swazi',
    'Swaziland': 'Swazi',
    'Gabon': 'Gabonese',
    'Gambia': 'Gambian',
    'Guinea': 'Guinean',
    'Guinea-Bissau': 'Guinea-Bissauan',
    'Guyana': 'Guyanese',
    'Kyrgyzstan': 'Kyrgyz',
    'Lesotho': 'Mosotho',
    'Liberia': 'Liberian',
    'Maldives': 'Maldivian',
    'Mali': 'Malian',
    'Malawi': 'Malawian',
    'Mauritania': 'Mauritanian',
    'Moldova': 'Moldovan',
    'Namibia': 'Namibian',
    'Niger': 'Nigerien',
    'North Macedonia': 'Macedonian',
    'Macedonia': 'Macedonian',
    'Rwanda': 'Rwandan',
    'Sierra Leone': 'Sierra Leonean',
    'Somalia': 'Somali',
    'South Sudan': 'South Sudanese',
    'Suriname': 'Surinamese',
    'Tajikistan': 'Tajik',
    'Togo': 'Togolese',
    'Turkmenistan': 'Turkmen',
    'Zambia': 'Zambian',
    'Kosovo': 'Kosovar',
    'Montenegro': 'Montenegrin',
    'Seychelles': 'Seychellois',
    'Timor-Leste': 'Timorese',
    'East Timor': 'Timorese',
    'Tonga': 'Tongan',
    'Vanuatu': 'Ni-Vanuatu',
    'Samoa': 'Samoan',
    'Solomon Islands': 'Solomon Islander',
    'Marshall Islands': 'Marshallese',
    'Micronesia': 'Micronesian',
    'Palau': 'Palauan',
    'Kiribati': 'I-Kiribati',
    'Tuvalu': 'Tuvaluan',
    'Nauru': 'Nauruan',
    
    # Country name variations from dataset
    'Hong Kong SAR of China': 'Hong Kong',
    'Hong Kong': 'Hong Kong',
    'Lao PDR': 'Laotian',
    'Laos': 'Laotian',
    'Myanmar': 'Myanmar',
    'Burma': 'Myanmar',
    'North Cyprus': 'Cypriot',
    'Puerto Rico': 'Puerto Rican',
    'Republic of Korea': 'South Korean',
    'South Korea': 'South Korean',
    'Korea': 'Korean',
    'Republic of Moldova': 'Moldovan',
    'Moldova': 'Moldovan',
    'Russian Federation': 'Russian',
    'Russia': 'Russian',
    'Somaliland Region': 'Somali',
    'Somalia': 'Somali',
    'State of Palestine': 'Palestinian',
    'Palestine': 'Palestinian',
    'Taiwan Province of China': 'Taiwanese',
    'Taiwan': 'Taiwanese',
    'Türkiye': 'Turkish',
    'Turkey': 'Turkish',
    'Viet Nam': 'Vietnamese',
    'Vietnam': 'Vietnamese',
}

def country_to_nationality(country_name):
    """
    Convert country name to nationality adjective
    
    Args:
        country_name: Name of the country (e.g., "United States", "Finland")
    
    Returns:
        Nationality adjective (e.g., "American", "Finnish")
    """
    # Direct lookup
    if country_name in COUNTRY_TO_NATIONALITY:
        return COUNTRY_TO_NATIONALITY[country_name]
    
    # Try case-insensitive lookup
    country_lower = country_name.lower()
    for country, nationality in COUNTRY_TO_NATIONALITY.items():
        if country.lower() == country_lower:
            return nationality
    
    # Special cases that might not match exactly
    if 'Côte' in country_name or 'Cote' in country_name:
        return 'Ivorian'
    if country_name == 'Myanmar' or country_name == 'Burma':
        return 'Myanmar'
    
    # If not found, try common patterns
    # Countries ending in -ia often become -ian
    if country_name.endswith('ia') and len(country_name) > 4:
        return country_name[:-2] + 'ian'
    
    # Countries ending in -land often become -ish or -er
    if country_name.endswith('land'):
        if country_name == 'Iceland':
            return 'Icelandic'
        elif country_name == 'Finland':
            return 'Finnish'
        elif country_name == 'Poland':
            return 'Polish'
        elif country_name == 'Ireland':
            return 'Irish'
        else:
            return country_name[:-4] + 'er'
    
    # Countries ending in -stan often become -i
    if country_name.endswith('stan'):
        base = country_name[:-4]
        if base == 'Afghan':
            return 'Afghan'
        elif base == 'Kazakh':
            return 'Kazakh'
        elif base == 'Uzbek':
            return 'Uzbek'
        else:
            return base + 'i'
    
    # Countries ending in -y often become -ian or -ese
    if country_name.endswith('y'):
        if country_name == 'Turkey':
            return 'Turkish'
        elif country_name == 'Italy':
            return 'Italian'
        elif country_name == 'Germany':
            return 'German'
        else:
            return country_name[:-1] + 'ian'
    
    # Default: return country name with common suffix
    # This is a fallback - should be improved with more rules
    return country_name

def get_nationality_for_countries(countries):
    """
    Convert a list of country names to nationalities
    
    Args:
        countries: List of country names
    
    Returns:
        Dictionary mapping country names to nationalities
    """
    return {country: country_to_nationality(country) for country in countries}

