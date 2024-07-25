'''
'''

import numpy as np

fbc_wind_speed = {
    '100': 1,
    '110': 2, 
    '>=120': 3
}

zipcode = {}
address_county = {}
census_block_group = {}
roof_type = {}

address_county = {
    'Desoto': 'DeSoto',
    'Saint Johns': 'St. Johns',
    'Saint Lucie': 'St. Lucie'
}

protection_class = {
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '8B': '8',  # NOTE: think this should be 8 since no half class exists
    '9': '9',
    '10': '10'
}  

construction_type = {
    'Frame': 'Frame',
    'Masonry': 'Masonry', 
    'Concrete': 'Concrete', 
    'Superior - Fire Resistive': 'Superior - Fire Resistive',
    'Wood Frame': 'Frame'
}

exterior_wall_finish = {
    'Siding - Vinyl': 'Siding', 
    'Traditional Stucco': 'Stucco', 
    'Stone Veneer (Natural)': 'Veneer',
    'Siding - Hardboard/Masonite': 'Siding', 
    'Siding - Pine (Clapboard)': 'Siding',
    'Brick Veneer': 'Veneer', 
    'Siding - Alum. or Metal': 'Siding',
    'None - Included In Ext. Wall Construction': 'None',
    'Siding - Cedar (Clapboard)': 'Siding', 
    'Synthetic Stucco': 'Stucco',
    'Cement Fiber (Shingle)': 'Cement', 
    'Brick - Solid': 'Brick',
    'Siding - Plywood (Vertical Groove)': 'Siding',
    'Concrete Block - Decorative': 'Veneer',
    'Siding - Cement Fiber (Clapboard)': 'Siding',
    'Siding - Cedar (Tongue & Groove)': 'Siding',
    'Siding - Board and Batten': 'Siding',
    'Wood Shingle/Shake': 'Wood', 
    'Brick Veneer - Custom': 'Veneer',
    'Siding - Vinyl Shingles': 'Siding', 
    'Siding - Redwood (Clapboard)': 'Siding',
    'Siding - Redwood (Tongue & Groove)': 'Siding', 
    'Brick - Solid - Custom': 'Brick',
    'Siding - Pine (Tongue & Groove)': 'Siding', 
    'Metal - Corrugated Galvanized': 'Metal',
    'Cypress - Reclaimed': 'Wood', 
    'Metal - Copper Shingle': 'Metal',
    'Stone Veneer (Manufactured)': 'Veneer', 
    'Solid Stone': 'Stone', 
    'Masonry Stucco': 'Stucco',
    'Siding - Steel': 'Siding', 
    'Metal - Painted Ribbed': 'Metal'
}

bceg = {
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    'Ungraded': '0',
    'Non-Participating': '0'
}

roof_shape = {
    'Other': 'Other', 
    'Hip': 'Hip', 
    'Gable': 'Gable', 
    'Complex/Custom': 'Other', 
    'Flat': 'Flat', 
    'Mansard': 'Other', 
    'Gambrel': 'Other',
    'Shed': 'Other'
}

foundation = {
    'Slab': 'Slab',
    'Crawlspace': 'Crawlspace',
    'Finished Basement': 'Basement',
    'Partially Finished Basement': 'Basement',
    'Unfinished Basement': 'Basement',
    'Elevated Slab': 'Elevated',
    '': 'Slab',
    'Elevated Piers/Pilings':'Elevated',
    'Elevated with Enclosure':'Elevated',
    'Split Level': 'Basement'
}

payment_schedule = {
    'full': 'annual', 
    'two_pay': 'semi-annual', 
    'quarterly': 'quarterly', 
    '1 Pay': 'annual'
}

property_type = {
    'Single Family Detached': 'Single Family',
    'Townhouse': 'Townhome',
    'Single Family Attached End Unit': 'Townhome', #
    'Single Family Attached Interior Unit': 'Townhome', #
    'Two Family (Duplex)': 'Multi-Family', #
    '': 'Other', #
    'Other': 'Other', #
    'Four Family (Four-plex)': 'Multi-Family', #
    'Single-Family Home': 'Single Family',
    'Multi-family (unknown # of units)': 'Multi-Family', #
    'Rowhouse': 'Townhome', #
    'Three Family (Tri-plex)': 'Multi-Family'}

line = {'ho3': 'ho3'}

state = {'FL': 'FL'}

occupancy = {'Owner Occupied': 'Owner Occupied'}

roof_cover = {
    'non_fbc_equivalent': 'non_fbc_equivalent', 
    'fbc_equivalent': 'fbc_equivalent',
    'reinforced_concrete_roof_deck': 'reinforced_concrete_roof_deck'
}

roof_deck = {
    'other': 'other', 
    'reinforced_concrete_roof_deck': 'reinforced_concrete_roof_deck'
}

roof_deck_attachment = {
    '6d@6\\"/12\\"': '6d@6\\"/12\\"',
    '8d@6\\"/12\\"': '8d@6\\"/12\\"',
    '8d@6\\"/6\\"': '8d@6\\"/6\\"',
    'unknown': '6d@6"/12"',
    'no_attic_access': '6d@6"/12"', 
    'other': '6d@6"/12"', 
    'reinforced_concrete_roof_deck': 'reinforced_concrete_roof_deck'
}

roof_wall_connection = {
    'toe_nails': 'toe_nails', 
    'single_wraps': 'single_wraps', 
    'clips': 'clips', 
    'double_wraps': 'double_wraps',
    'reinforced_concrete_roof_deck': 'reinforced_concrete_roof_deck'
}

opening_protection = {
    'none': 'none', 
    'hurricane': 'hurricane', 
    'basic': 'basic'
}

terrain = {
    'b': 'b', 
    'c': 'c', 
    'hvhz': 'hvhz'
}

builder = {'Standard': 'Standard'}

burglar_alarm = {
    'none': 'none', 
    'central': 'central', 
    'local': 'other', 
    'direct': 'other', 
    'smart_burglar_alarm': 'other',
    'unspecified/uncategorized': 'other', 
    'motion_detecting_camera': 'other'
}

fire_alarm_monitoring = {
    'none': 'none', 
    'central': 'central', 
    'local': 'local', 
    'direct': 'direct'
}

fire_alarm_sprinkler = {
    'none': 'none', 
    'partial': 'partial', 
    'full': 'full', 
    'class_a': 'class_a'
}

secured_community = {
    'no': 'no', 
    'gated_key_card': 'gated_key_card', 
    'gated_manned_guard': 'gated_manned_guard', 
    'single_entry': 'single_entry',
    'community_patrol': 'community_patrol'
}

water_damage = {
    'full': 'full',
    'excluded': 'excluded', 
    'limited': 'limited'
}

payment_type = {
    'escrow': 'escrow', 
    'card': 'card', 
    'ach': np.nan
}

loss_settlement = {'replacement cost': 'replacement cost'}

liability_medical_payments = {
    '$300,000 / $5,000': '$300,000 / $5,000', 
    '$500,000 / $5,000': '$500,000 / $5,000', 
    '$0 / $0': '$0 / $0',
    '$100,000 / $1,000': '$100,000 / $1,000'
}

prior_liability_limit = {'$300,000 or Greater': '$300,000 or Greater'}

usage = {
    'Primary': 'Primary', 
    'Seasonal': 'Seasonal', 
    'Secondary': 'Seasonal'
}

flood_zone = {
    'X': 'X', 
    'A': 'A', 
    'AE': 'AE', 
    'AH': 'AH', 
    'AO': 'AO', 
    'VE': 'VE', 
    'D': 'D'
}

wildfire_grade = {
    'A': 'A', 
    'B': 'B', 
    'C': 'C', 
    'D': 'D'
}

WUICLASS2020 = {
    'Med_Dens_NoVeg': 'Med_Dens_NoVeg',
    'Med_Dens_Interface': 'Med_Dens_Interface',
    'High_Dens_NoVeg': 'High_Dens_NoVeg',
    'High_Dens_Interface': 'High_Dens_Interface',
    'Med_Dens_Intermix': 'Med_Dens_Intermix',
    'Low_Dens_Intermix': 'Low_Dens_Intermix',
    'Low_Dens_NoVeg': 'Low_Dens_NoVeg',
    'Low_Dens_Interface': 'Low_Dens_Interface',
    'Uninhabited_NoVeg': 'Uninhabited_NoVeg',
    'Very_Low_Dens_Veg': 'Very_Low_Dens_Veg',
    'Very_Low_Dens_NoVeg': 'Very_Low_Dens_NoVeg',
    'Water': 'Water',
    'Uninhabited_Veg': 'Uninhabited_Veg',
    'High_Dens_Intermix': 'Uninhabited_Veg'
}

roof_type2 = {
    'Membrane - EPDM or PVC': ('Superior w/ Poured Concrete', 'Built-Up'),
    'Synthetic Composite Roofing': ('Composition Shingle', 'Shingle'),
    'Slate': ('Slate', 'Tile'),
    'Wood Shingles or Shakes': ('Wood Shingle', 'Wood'),
    'Composition - 3 Tab Shingle': ('Composition Shingle', 'Shingle'),
    'Tile - Clay': ('Tile', 'Tile'),
    'Metal - Tile/Shake': ('Metal', 'Metal'),
    'Built-up (hot mopped) w/Gravel': ('Tar & Gravel', 'Built-Up'),
    'Composition - Architectural Shingle': ('Shingles, Architectural', 'Shingle'),
    'Tile - Concrete': ('Concrete Tile', 'Tile'),
    'Composition - Roll Roofing': ('Rolled Roof', 'Shingle'),
    'Metal - Corrugated Galvanized': ('Aluminum Corrugated', 'Metal'),
    'Metal - Standing Seam': ('Standing Seam Metal', 'Metal'),
    'Metal - Painted Rib': ('Metal', 'Metal'),
    'Tile - Cement Fiber': ('Concrete Tile', 'Tile'),
    'Tile - Glazed': ('Tile', 'Tile'),
    'Composition - Impact Resist. Shingle': ('Shingles, Architectural', 'Shingle'),
    'Built-up (hot mopped) w/o Gravel': ('Other', 'Built-Up'),
    'Metal - Copper Shingle': ('Metal', 'Metal'),
    'Metal - Standing Seam Copper': ('Standing Seam Metal', 'Metal'),
    'Wood Shingles/Shakes-Deco Ptrn.': ('Wood Shingle', 'Wood'),
    'Sprayed Polyurethane Foam (SPF)': ('Other', 'Built-Up'),
    'NULL': ('Composition Shingle', 'Shingle'),
    '': ('Composition Shingle', 'Shingle')
}

pricing_construction = {
    'FrameTraditional Stucco':'Frame',
    'FrameBrick Veneer':'Masonry Veneer',
    'FrameSiding - Cedar (Clapboard)':'Frame',
    'FrameNone - Included In Ext. Wall Construction':'Frame',
    'FrameSynthetic Stucco':'Frame',
    'FrameSiding - Vinyl':'Frame',
    'FrameSiding - Alum. or Metal':'Frame',
    'FrameMasonry Stucco':'Frame',
    'FrameSiding - Cement Fiber (Clapboard)':'Frame with Hardiplank Siding',
    'FrameStone Veneer (Natural)':'Masonry Veneer',
    'FrameConcrete Block - Decorative':'Masonry Veneer',
    'FrameSiding - Plywood (Vertical Groove)':'Frame',
    'FrameCement Fiber (Shingle)':'Frame with Hardiplank Siding',
    'FrameWood Shingle/Shake':'Frame',
    'FrameSiding - Hardboard/Masonite':'Frame with Hardiplank Siding',
    'FrameSiding - Board and Batten':'Frame',
    'FrameSiding - Cedar (Tongue & Groove)':'Frame',
    'FrameBrick - Solid':'Masonry Veneer',
    'FrameSiding - Pine (Clapboard)':'Frame',
    'FrameSiding - Vinyl Shingles':'Frame',
    'FrameSiding - Pine (Tongue & Groove)':'Frame',
    'Frame':'Frame',
    'FrameStone Veneer (Manufactured)':'Masonry Veneer',
    'FrameSiding - Redwood (Clapboard)':'Frame',
    'FrameBrick Veneer - Custom':'Masonry Veneer',
    'FrameSolid Stone':'Masonry Veneer',
    'FrameMetal - Corrugated Galvanized':'Frame',
    'FrameCypress - Reclaimed':'Frame',
    'FrameMetal - Copper Shingle':'Frame',
    'FrameBrick - Solid - Custom':'Masonry Veneer',
    'FrameMetal - Painted Ribbed':'Frame',
    'FrameSiding - Steel':'Frame',
    'FrameSiding - Redwood (Tongue & Groove)':'Frame',
    'MasonryTraditional Stucco':'Masonry',
    'MasonryBrick Veneer':'Masonry',
    'MasonrySiding - Cedar (Clapboard)':'Masonry',
    'MasonryNone - Included In Ext. Wall Construction':'Masonry',
    'MasonrySynthetic Stucco':'Masonry',
    'MasonrySiding - Vinyl':'Masonry',
    'MasonrySiding - Alum. or Metal':'Masonry',
    'MasonryMasonry Stucco':'Masonry',
    'MasonrySiding - Cement Fiber (Clapboard)':'Masonry',
    'MasonryStone Veneer (Natural)':'Masonry',
    'MasonryConcrete Block - Decorative':'Masonry',
    'MasonrySiding - Plywood (Vertical Groove)':'Masonry',
    'MasonryCement Fiber (Shingle)':'Masonry',
    'MasonryWood Shingle/Shake':'Masonry',
    'MasonrySiding - Hardboard/Masonite':'Masonry',
    'MasonrySiding - Board and Batten':'Masonry',
    'MasonrySiding - Cedar (Tongue & Groove)':'Masonry',
    'MasonryBrick - Solid':'Masonry',
    'MasonrySiding - Pine (Clapboard)':'Masonry',
    'MasonrySiding - Vinyl Shingles':'Masonry',
    'MasonrySiding - Pine (Tongue & Groove)':'Masonry',
    'Masonry':'Masonry',
    'MasonryStone Veneer (Manufactured)':'Masonry',
    'MasonrySiding - Redwood (Clapboard)':'Masonry',
    'MasonryBrick Veneer - Custom':'Masonry',
    'MasonrySolid Stone':'Masonry',
    'MasonryMetal - Corrugated Galvanized':'Masonry',
    'MasonryCypress - Reclaimed':'Masonry',
    'MasonryMetal - Copper Shingle':'Masonry',
    'MasonryBrick - Solid - Custom':'Masonry',
    'MasonryMetal - Painted Ribbed':'Masonry',
    'MasonrySiding - Steel':'Masonry',
    'MasonrySiding - Redwood (Tongue & Groove)':'Masonry',
    'ConcreteTraditional Stucco':'Masonry',
    'ConcreteBrick Veneer':'Masonry',
    'ConcreteSiding - Cedar (Clapboard)':'Masonry',
    'ConcreteNone - Included In Ext. Wall Construction':'Masonry',
    'ConcreteSynthetic Stucco':'Masonry',
    'ConcreteSiding - Vinyl':'Masonry',
    'ConcreteSiding - Alum. or Metal':'Masonry',
    'ConcreteMasonry Stucco':'Masonry',
    'ConcreteSiding - Cement Fiber (Clapboard)':'Masonry',
    'ConcreteStone Veneer (Natural)':'Masonry',
    'ConcreteConcrete Block - Decorative':'Masonry',
    'ConcreteSiding - Plywood (Vertical Groove)':'Masonry',
    'ConcreteCement Fiber (Shingle)':'Masonry',
    'ConcreteWood Shingle/Shake':'Masonry',
    'ConcreteSiding - Hardboard/Masonite':'Masonry',
    'ConcreteSiding - Board and Batten':'Masonry',
    'ConcreteSiding - Cedar (Tongue & Groove)':'Masonry',
    'ConcreteBrick - Solid':'Masonry',
    'ConcreteSiding - Pine (Clapboard)':'Masonry',
    'ConcreteSiding - Vinyl Shingles':'Masonry',
    'ConcreteSiding - Pine (Tongue & Groove)':'Masonry',
    'Concrete':'Masonry',
    'ConcreteStone Veneer (Manufactured)':'Masonry',
    'ConcreteSiding - Redwood (Clapboard)':'Masonry',
    'ConcreteBrick Veneer - Custom':'Masonry',
    'ConcreteSolid Stone':'Masonry',
    'ConcreteMetal - Corrugated Galvanized':'Masonry',
    'ConcreteCypress - Reclaimed':'Masonry',
    'ConcreteMetal - Copper Shingle':'Masonry',
    'ConcreteBrick - Solid - Custom':'Masonry',
    'ConcreteMetal - Painted Ribbed':'Masonry',
    'ConcreteSiding - Steel':'Masonry',
    'ConcreteSiding - Redwood (Tongue & Groove)':'Masonry',
    'Superior - Fire ResistiveTraditional Stucco':'Superior - Fire Resistive',
    'Superior - Fire ResistiveBrick Veneer':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Cedar (Clapboard)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveNone - Included In Ext. Wall Construction':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSynthetic Stucco':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Vinyl':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Alum. or Metal':'Superior - Fire Resistive',
    'Superior - Fire ResistiveMasonry Stucco':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Cement Fiber (Clapboard)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Plywood (Vertical Groove)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveCement Fiber (Shingle)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveWood Shingle/Shake':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Hardboard/Masonite':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Board and Batten':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Cedar (Tongue & Groove)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveBrick - Solid':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Pine (Clapboard)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Vinyl Shingles':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Pine (Tongue & Groove)':'Superior - Fire Resistive',
    'Superior - Fire Resistive':'Superior - Fire Resistive',
    'Superior - Fire ResistiveStone Veneer (Manufactured)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Redwood (Clapboard)':'Superior - Fire Resistive',
    'Superior - Fire ResistiveBrick Veneer - Custom':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSolid Stone':'Superior - Fire Resistive',
    'Superior - Fire ResistiveMetal - Corrugated Galvanized':'Superior - Fire Resistive',
    'Superior - Fire ResistiveCypress - Reclaimed':'Superior - Fire Resistive',
    'Superior - Fire ResistiveMetal - Copper Shingle':'Superior - Fire Resistive',
    'Superior - Fire ResistiveBrick - Solid - Custom':'Superior - Fire Resistive',
    'Superior - Fire ResistiveMetal - Painted Ribbed':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Steel':'Superior - Fire Resistive',
    'Superior - Fire ResistiveSiding - Redwood (Tongue & Groove)':'Superior - Fire Resistive',
    'Wood FrameTraditional Stucco':'Frame',
    'Wood FrameBrick Veneer':'Masonry Veneer',
    'Wood FrameSiding - Cedar (Clapboard)':'Frame',
    'Wood FrameNone - Included In Ext. Wall Construction':'Frame',
    'Wood FrameSynthetic Stucco':'Frame',
    'Wood FrameSiding - Vinyl':'Frame',
    'Wood FrameSiding - Alum. or Metal':'Frame',
    'Wood FrameMasonry Stucco':'Frame',
    'Wood FrameSiding - Cement Fiber (Clapboard)':'Frame with Hardiplank Siding',
    'Wood FrameStone Veneer (Natural)':'Masonry Veneer',
    'Wood FrameConcrete Block - Decorative':'Masonry Veneer',
    'Wood FrameSiding - Plywood (Vertical Groove)':'Frame',
    'Wood FrameCement Fiber (Shingle)':'Frame with Hardiplank Siding',
    'Wood FrameWood Shingle/Shake':'Frame',
    'Wood FrameSiding - Hardboard/Masonite':'Frame with Hardiplank Siding',
    'Wood FrameSiding - Board and Batten':'Frame',
    'Wood FrameSiding - Cedar (Tongue & Groove)':'Frame',
    'Wood FrameBrick - Solid':'Masonry Veneer',
    'Wood FrameSiding - Pine (Clapboard)':'Frame',
    'Wood FrameSiding - Vinyl Shingles':'Frame',
    'Wood FrameSiding - Pine (Tongue & Groove)':'Frame',
    'Wood Frame':'Frame',
    'Wood FrameStone Veneer (Manufactured)':'Masonry Veneer',
    'Wood FrameSiding - Redwood (Clapboard)':'Frame',
    'Wood FrameBrick Veneer - Custom':'Masonry Veneer',
    'Wood FrameSolid Stone':'Masonry Veneer',
    'Wood FrameMetal - Corrugated Galvanized':'Frame',
    'Wood FrameCypress - Reclaimed':'Frame',
    'Wood FrameMetal - Copper Shingle':'Frame',
    'Wood FrameBrick - Solid - Custom':'Masonry Veneer',
    'Wood FrameMetal - Painted Ribbed':'Frame',
    'Wood FrameSiding - Steel':'Frame',
    'Wood FrameSiding - Redwood (Tongue & Groove)':'Frame',
    '':np.nan
}

pricing_sprinkler = {
    'centralclass_a':'Central and Class A Sprinkler',
    'centralclass_b':'Central and Class B Sprinkler',
    'centralnone':'Central',
    'noneclass_a':'Class A Sprinkler',
    'noneclass_b':'Class B Sprinkler',
    'nonenone':'None',
    'central':'Central',
    'central0':'Central',
    '00':'None',
    '0class_a':'Class A Sprinkler',
    '0class_b':'Class B Sprinkler',
    '0none':'None',
    '0none':'None',
    'NULLNULL':'None',
    'NULLclass_a':'Class A Sprinkler',
    'NULLclass_b':'Class B Sprinkler',
    'NULLnone':'None',
    'centralNULL':'Central',
    'noneNULL':'None',
    'centralpartial':'Central and Class A Sprinkler',
    'centralfull':'Central and Class B Sprinkler',
    'nonepartial':'Class A Sprinkler',
    'nonefull':'Class B Sprinkler',
    '0partial':'Class A Sprinkler',
    '0full':'Class B Sprinkler',
    'NULLpartial':'Class A Sprinkler',
    'NULLfull':'Class B Sprinkler',
    'localnone':'None',
    'localNULL':'None',
    'directnone':'Fire',
    'directNULL':'Fire',
    'none':'None',
    'none':'None',
    '':'None',
    'local':'None',
    'direct':'None',
    'full':'Class B Sprinkler',
    'partial':'Class A Sprinkler',
    'localfull':'Class B Sprinkler',
    'localpartial':'Class A Sprinkler',
    'directpartial':'Class A Sprinkler',
}

pricing_water_detection = {
    'FALSEFALSE':'None',
    'TRUEFALSE':'Auto Shut off Valve',
    'nofalse':'None',
    'false':'None',
    'FALSETRUE':'None',
    'TRUETRUE':'Auto Shut off Valve with Central Alarm',
    'notrue':'None',
    'false':'None',
    'true':'Auto Shut off Valve',
    'no':'None',
    '':'None',
    False: 'None',
    True: 'Auto Shut off Valve'
}

roof_type = {
    'Composition - 3 Tab Shingle': 'Composition - 3 Tab Shingle',
    'Composition - Architectural Shingle': 'Composition - Architectural Shingle',
    'Tile - Clay': 'Tile - Clay',
    'Metal - Standing Seam': 'Metal - Standing Seam',
    'Tile - Concrete': 'Tile - Concrete',
    'Built-up (hot mopped) w/Gravel': 'Built-up (hot mopped) w/Gravel',
    'Composition - Roll Roofing': 'Composition - Roll Roofing',
    'Metal - Tile/Shake': 'Metal - Tile/Shake',
    'Metal - Painted Rib': 'Metal - Painted Rib',
    'Membrane - EPDM or PVC': 'Membrane - EPDM or PVC',
    'Synthetic Composite Roofing': 'Synthetic Composite Roofing',
    'Metal - Standing Seam Copper': 'Metal - Standing Seam Copper',
    'Built-up (hot mopped) w/o Gravel': 'Built-up (hot mopped) w/o Gravel',
    'Metal - Corrugated Galvanized': 'Metal - Corrugated Galvanized',
    'Wood Shingles or Shakes': 'Wood Shingles or Shakes',
    'Sprayed Polyurethane Foam (SPF)': 'Sprayed Polyurethane Foam (SPF)',
    'Tile - Cement Fiber': 'Tile - Cement Fiber',
    'Slate': 'Slate',
    'Composition - Impact Resist. Shingle': 'Composition - Impact Resist. Shingle',
    'Metal - Copper Shingle': 'Metal - Copper Shingle'
}

WUICLASS2020 = {
    'Med_Dens_NoVeg': 'Med_Dens_NoVeg',
    'Med_Dens_Interface': 'Med_Dens_Interface',
    'High_Dens_NoVeg': 'High_Dens_NoVeg',
    'High_Dens_Interface': 'High_Dens_Interface',
    'Med_Dens_Intermix': 'Med_Dens_Intermix',
    'Low_Dens_Intermix': 'Low_Dens_Intermix',
    'Low_Dens_NoVeg': 'Low_Dens_NoVeg',
    'Low_Dens_Interface': 'Low_Dens_Interface',
    'Uninhabited_NoVeg': 'Uninhabited_NoVeg',
    'Very_Low_Dens_Veg': 'Very_Low_Dens_Veg',
    'Very_Low_Dens_NoVeg': 'Very_Low_Dens_NoVeg',
    'Uninhabited_Veg': 'Uninhabited_Veg',
    'Water': 'Water',
    'High_Dens_Intermix': 'High_Dens_Intermix'
}


def get_category_map(column):
    '''
    Parameters
    ----------
    column : str
        Column name from observations dataframe.

    Returns
    -------
    maps(column) : dicitonary
        Dictionary contains the current and new values for that column.
    '''
    
    maps = {'line': line,
             'state': state,
             'zipcode': zipcode,
             'address_county': address_county,
             'census_block_group': census_block_group,
             'protection_class': protection_class,
             'construction_type': construction_type,
             'exterior_wall_finish': exterior_wall_finish,
             'occupancy': occupancy,
             'bceg': bceg,
             'roof_shape': roof_shape,
             'roof_type': roof_type,
             'roof_type2': roof_type2,
             'roof_cover': roof_cover,
             'roof_deck': roof_deck,
             'roof_deck_attachment': roof_deck_attachment,
             'roof_wall_connection': roof_wall_connection,
             'opening_protection': opening_protection,
             'terrain': terrain,
             'foundation': foundation,
             'builder': builder,
             'burglar_alarm': burglar_alarm,
             'fire_alarm_monitoring': fire_alarm_monitoring,
             'fire_alarm_sprinkler': fire_alarm_sprinkler,
             'secured_community': secured_community,
             'water_damage': water_damage,
             'payment_schedule': payment_schedule,
             'payment_type': payment_type,
             'property_type': property_type,
             'loss_settlement': loss_settlement,
             'liability_medical_payments': liability_medical_payments,
             'prior_liability_limit': prior_liability_limit,
             'usage': usage,
             'flood_zone': flood_zone,
             'wildfire_grade': wildfire_grade,
             'pricing_construction': pricing_construction,
             'pricing_sprinkler': pricing_sprinkler,
             'pricing_water_detection': pricing_water_detection,
             'WUICLASS2020': WUICLASS2020,
             'fbc_wind_speed': fbc_wind_speed,
             'address_county': address_county
           }
    
    return maps[column]