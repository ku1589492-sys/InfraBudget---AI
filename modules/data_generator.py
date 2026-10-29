"""
Multi-State & District Municipal Data Engine for InfraBudget AI.
Comprehensive geographic hierarchy: All 29 States & UTs -> District / Municipal Corporation -> Wards & Villages.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

STATE_DISTRICT_DATABASE = {
    "Andhra Pradesh": {
        "districts": {
            "Visakhapatnam (GVMC)": {
                "full_name": "Greater Visakhapatnam Municipal Corporation (GVMC)",
                "center_lat": 17.6868,
                "center_lon": 83.2185,
                "wards": [
                    {"id": 1, "name": "Ward 01 - MVP Colony", "village": "MVP Colony Urban Village", "zone": "Zone 2", "lat": 17.7420, "lon": 83.3370, "type": "Planned Coastal Residential"},
                    {"id": 2, "name": "Ward 02 - Gajuwaka Industrial", "village": "Gajuwaka Industrial Village", "zone": "Zone 5", "lat": 17.6930, "lon": 83.2140, "type": "Heavy Steel & Petrochemical Hub"},
                    {"id": 3, "name": "Ward 03 - Madhurawada IT Hub", "village": "Madhurawada Revenue Village", "zone": "Zone 1", "lat": 17.8100, "lon": 83.3550, "type": "Rapid Expansion IT Corridor"},
                    {"id": 4, "name": "Ward 04 - Pendurthi Outskirts", "village": "Pendurthi Rural Village", "zone": "Zone 6", "lat": 17.8340, "lon": 83.2030, "type": "Peri-Urban Agricultural Influx"},
                    {"id": 5, "name": "Ward 05 - Bheemunipatnam", "village": "Bheemili Dutch Heritage Village", "zone": "Zone 1", "lat": 17.8920, "lon": 83.4540, "type": "Coastal Fishery & Tourism"},
                    {"id": 6, "name": "Ward 06 - Jagadamba Junction", "village": "Daba Gardens Mohalla", "zone": "Zone 3", "lat": 17.7120, "lon": 83.3010, "type": "Dense Commercial Wholesale Core"}
                ]
            },
            "Vijayawada (VMC)": {
                "full_name": "Vijayawada Municipal Corporation (VMC)",
                "center_lat": 16.5062,
                "center_lon": 80.6480,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Benz Circle", "village": "Patamata Urban Village", "zone": "Circle 2", "lat": 16.4980, "lon": 80.6550, "type": "Prime Commercial Corridor"},
                    {"id": 2, "name": "Ward 02 - Bhavanipuram", "village": "Bhavanipuram River Village", "zone": "Circle 1", "lat": 16.5240, "lon": 80.5960, "type": "Krishna River Floodplain Dense"},
                    {"id": 3, "name": "Ward 03 - Gunadala Hill", "village": "Gunadala Hillock Village", "zone": "Circle 3", "lat": 16.5180, "lon": 80.6690, "type": "Undulating Terrain Drainage Deficit"},
                    {"id": 4, "name": "Ward 04 - Gollapudi Wholesale", "village": "Gollapudi Agri Village", "zone": "Circle 1", "lat": 16.5510, "lon": 80.5820, "type": "Agri Logistics Freight Hub"},
                    {"id": 5, "name": "Ward 05 - Autonagar Light Industrial", "village": "Kanuru Village Boundary", "zone": "Circle 2", "lat": 16.4880, "lon": 80.6820, "type": "Automotive Workshops Cluster"}
                ]
            }
        }
    },
    "Arunachal Pradesh": {
        "districts": {
            "Itanagar Capital Complex (IMC)": {
                "full_name": "Itanagar Municipal Corporation (IMC)",
                "center_lat": 27.0844,
                "center_lon": 93.6053,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Ganga Market", "village": "Ganga Old Village", "zone": "Central", "lat": 27.0910, "lon": 93.6180, "type": "Hilly Commercial Core"},
                    {"id": 2, "name": "Ward 02 - Naharlagun Sub-City", "village": "Model Village Naharlagun", "zone": "Transit Zone", "lat": 27.1060, "lon": 93.6930, "type": "Railway Terminal & Dense Transit"},
                    {"id": 3, "name": "Ward 03 - Chimpu Foothills", "village": "Chimpu Tribal Village", "zone": "South", "lat": 27.0620, "lon": 93.5950, "type": "Mountain Slope Soil Erosion Risk"},
                    {"id": 4, "name": "Ward 04 - Nirjuli University Corridor", "village": "Nirjuli Gaon", "zone": "East", "lat": 27.1280, "lon": 93.7440, "type": "Institutional & High Youth Footfall"},
                    {"id": 5, "name": "Ward 05 - Banderdewa Checkpost", "village": "Banderdewa Border Basti", "zone": "North-East", "lat": 27.1480, "lon": 93.8210, "type": "State Border Cargo Transit"}
                ]
            }
        }
    },
    "Assam": {
        "districts": {
            "Guwahati (GMC)": {
                "full_name": "Guwahati Municipal Corporation (GMC)",
                "center_lat": 26.1445,
                "center_lon": 91.7362,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Dispur Secretariat", "village": "Dispur Capital Village", "zone": "Zone 4", "lat": 26.1400, "lon": 91.7900, "type": "Administrative / Institutional"},
                    {"id": 2, "name": "Ward 02 - Paltan Bazaar Core", "village": "Ulubari Mohalla", "zone": "Zone 2", "lat": 26.1780, "lon": 91.7510, "type": "High Congestion Railway Choke"},
                    {"id": 3, "name": "Ward 03 - Jalukbari University", "village": "Jalukbari Gaon", "zone": "Zone 1", "lat": 26.1550, "lon": 91.6640, "type": "Brahmaputra Flood Plain Hub"},
                    {"id": 4, "name": "Ward 04 - Chandmari Hilltop", "village": "Chandmari Basti", "zone": "Zone 3", "lat": 26.1890, "lon": 91.7760, "type": "Landslide Vulnerable Slopes"},
                    {"id": 5, "name": "Ward 05 - Pan Bazaar Heritage", "village": "Fancy Bazaar Riverside", "zone": "Zone 2", "lat": 26.1850, "lon": 91.7450, "type": "Old City Wholesale Riverfront"},
                    {"id": 6, "name": "Ward 06 - Basistha Temple Catchment", "village": "Basistha Forest Village", "zone": "Zone 5", "lat": 26.0980, "lon": 91.7980, "type": "Eco-Sensitive Water Stream Zone"}
                ]
            },
            "Dibrugarh (DMB)": {
                "full_name": "Dibrugarh Municipal Board (DMB)",
                "center_lat": 27.4728,
                "center_lon": 94.9120,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Graham Bazaar", "village": "Graham Bazaar Mohalla", "zone": "North", "lat": 27.4760, "lon": 94.9080, "type": "River Dyke Vulnerable Zone"},
                    {"id": 2, "name": "Ward 02 - Boiragimoth", "village": "Boiragimoth Village", "zone": "Central", "lat": 27.4690, "lon": 94.9210, "type": "Planned Residential Sector"},
                    {"id": 3, "name": "Ward 03 - Milan Nagar Ext", "village": "Milan Nagar Gaon", "zone": "East", "lat": 27.4820, "lon": 94.9450, "type": "Suburban High Water Table"},
                    {"id": 4, "name": "Ward 04 - Amolapatty Historic", "village": "Old Amolapatty", "zone": "West", "lat": 27.4610, "lon": 94.9020, "type": "Traditional Heritage Basti"}
                ]
            }
        }
    },
    "Bihar": {
        "districts": {
            "Patna (PMC)": {
                "full_name": "Patna Municipal Corporation (PMC)",
                "center_lat": 25.5941,
                "center_lon": 85.1376,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Kankerbagh Colony", "village": "Kankerbagh Revenue Village", "zone": "Kankerbagh Zone", "lat": 25.5890, "lon": 85.1480, "type": "High Density Residential Grid"},
                    {"id": 2, "name": "Ward 02 - Danapur Cantt Corridor", "village": "Danapur Nizamat Gaon", "zone": "Danapur Zone", "lat": 25.6330, "lon": 85.0450, "type": "Peri-Urban Transit Outskirts"},
                    {"id": 3, "name": "Ward 03 - Digha Ghat Riverside", "village": "Digha Diara River Village", "zone": "Patliputra Zone", "lat": 25.6420, "lon": 85.0980, "type": "Ganga Inundation Buffer"},
                    {"id": 4, "name": "Ward 04 - Boring Road Retail", "village": "Sri Krishna Puri Locality", "zone": "New Capital Zone", "lat": 25.6170, "lon": 85.1140, "type": "Prime Commercial Corridor"},
                    {"id": 5, "name": "Ward 05 - Patna City Chowk", "village": "Begampur Historic Mohalla", "zone": "Patna City Zone", "lat": 25.5780, "lon": 85.2280, "type": "Extreme Density Traditional Basti"},
                    {"id": 6, "name": "Ward 06 - Phulwari Sharif", "village": "Phulwari Village Hub", "zone": "South Zone", "lat": 25.5720, "lon": 85.0820, "type": "Rapid Urbanizing Suburban Grid"}
                ]
            },
            "Gaya (GMC)": {
                "full_name": "Gaya Municipal Corporation (GMC)",
                "center_lat": 24.7914,
                "center_lon": 85.0002,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Bodhgaya Temple Core", "village": "Bodhgaya Heritage Village", "zone": "Temple Zone", "lat": 24.6960, "lon": 84.9910, "type": "UNESCO Heritage Tourism"},
                    {"id": 2, "name": "Ward 02 - Civil Lines Central", "village": "Civil Lines Ward Area", "zone": "Administrative", "lat": 24.7950, "lon": 85.0080, "type": "Government Offices Corridor"},
                    {"id": 3, "name": "Ward 03 - Manpur Handloom Cluster", "village": "Manpur Basti", "zone": "East", "lat": 24.8020, "lon": 85.0340, "type": "Falgu River Dry Bed Industrial"},
                    {"id": 4, "name": "Ward 04 - Delha Station Colony", "village": "Delha Railway Village", "zone": "West", "lat": 24.8140, "lon": 84.9890, "type": "Transit Interchange Dense"}
                ]
            }
        }
    },
    "Chhattisgarh": {
        "districts": {
            "Raipur (RMC)": {
                "full_name": "Raipur Municipal Corporation (RMC)",
                "center_lat": 21.2514,
                "center_lon": 81.6296,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Pandri Cloth Market", "village": "Pandri Village Locality", "zone": "Zone 2", "lat": 21.2640, "lon": 81.6510, "type": "Textile Wholesale Commercial"},
                    {"id": 2, "name": "Ward 02 - Telibandha Marine Drive", "village": "Telibandha Talab Area", "zone": "Zone 3", "lat": 21.2380, "lon": 81.6720, "type": "Urban Water Body Waterfront"},
                    {"id": 3, "name": "Ward 03 - Tatibandh Industrial", "village": "Tatibandh Rural Village", "zone": "Zone 1", "lat": 21.2610, "lon": 81.5640, "type": "AIIMS Logistics Heavy Corridor"},
                    {"id": 4, "name": "Ward 04 - Shankar Nagar VIP", "village": "Khamardih Gaon", "zone": "Zone 4", "lat": 21.2590, "lon": 81.6750, "type": "Planned High Income Sector"},
                    {"id": 5, "name": "Ward 05 - Gudhiyari Traditional", "village": "Gudhiyari Old Mohalla", "zone": "Zone 5", "lat": 21.2690, "lon": 81.6140, "type": "Railway Proximity Dense Core"}
                ]
            }
        }
    },
    "Goa": {
        "districts": {
            "North Goa (Panaji CCP)": {
                "full_name": "Corporation of the City of Panaji (CCP)",
                "center_lat": 15.4909,
                "center_lon": 73.8278,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Fontainhas Heritage", "village": "Fontainhas Latin Bairro", "zone": "Central", "lat": 15.4950, "lon": 73.8320, "type": "Portuguese Heritage Conservation"},
                    {"id": 2, "name": "Ward 02 - Miramar Beach Coast", "village": "Miramar Coastal Village", "zone": "West", "lat": 15.4820, "lon": 73.8050, "type": "Coastal High Tourism Area"},
                    {"id": 3, "name": "Ward 03 - Taleigao Suburban", "village": "Taleigao Agricultural Village", "zone": "South", "lat": 15.4640, "lon": 73.8190, "type": "Paddy Field Infill Rapid Growth"},
                    {"id": 4, "name": "Ward 04 - Ribandar Mandovi Bank", "village": "Ribandar River Village", "zone": "East", "lat": 15.5020, "lon": 73.8640, "type": "River Estuary Salinity Buffer"}
                ]
            }
        }
    },
    "Gujarat": {
        "districts": {
            "Ahmedabad (AMC)": {
                "full_name": "Ahmedabad Municipal Corporation (AMC)",
                "center_lat": 23.0225,
                "center_lon": 72.5714,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Navrangpura Central", "village": "Navrangpura Gaon", "zone": "West Zone", "lat": 23.0360, "lon": 72.5610, "type": "Commercial & University Core"},
                    {"id": 2, "name": "Ward 02 - Maninagar South", "village": "Maninagar Locality", "zone": "South Zone", "lat": 22.9980, "lon": 72.6020, "type": "Established High Density Core"},
                    {"id": 3, "name": "Ward 03 - Bopal Suburban Grid", "village": "Bopal Revenue Village", "zone": "New West Zone", "lat": 23.0350, "lon": 72.4640, "type": "Rapid Expansion Dense Grid"},
                    {"id": 4, "name": "Ward 04 - Ashram Road Sabarmati", "village": "Sabarmati Riverfront Village", "zone": "Central Zone", "lat": 23.0420, "lon": 72.5720, "type": "Riverfront Commercial Promenade"},
                    {"id": 5, "name": "Ward 05 - Naroda Industrial", "village": "Naroda GIDC Industrial Village", "zone": "North Zone", "lat": 23.0780, "lon": 72.6620, "type": "Heavy Chemical Effluent Stress"}
                ]
            },
            "Surat (SMC)": {
                "full_name": "Surat Municipal Corporation (SMC)",
                "center_lat": 21.1702,
                "center_lon": 72.8311,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Varachha Diamond Hub", "village": "Varachha Gaon Basti", "zone": "North Zone", "lat": 21.2140, "lon": 72.8680, "type": "Extreme Density Diamond Cutting"},
                    {"id": 2, "name": "Ward 02 - Athwa Lines Waterfront", "village": "Athwa Coastal Locality", "zone": "South-West Zone", "lat": 21.1680, "lon": 72.7930, "type": "Planned High Income Tapi Waterfront"},
                    {"id": 3, "name": "Ward 03 - Pandesara Industrial GIDC", "village": "Pandesara Rural Village", "zone": "South Zone", "lat": 21.1390, "lon": 72.8330, "type": "Textile Dyeing Effluent Buffer"},
                    {"id": 4, "name": "Ward 04 - Rander Historic Core", "village": "Rander Port Village", "zone": "West Zone", "lat": 21.2150, "lon": 72.7940, "type": "Historic Walled Town Congested"}
                ]
            }
        }
    },
    "Haryana": {
        "districts": {
            "Gurugram (MCG)": {
                "full_name": "Municipal Corporation of Gurugram (MCG)",
                "center_lat": 28.4595,
                "center_lon": 77.0266,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Cyber City & DLF", "village": "Nathupur Urban Village", "zone": "Zone 3", "lat": 28.4920, "lon": 77.0910, "type": "Global IT Tech Corridor"},
                    {"id": 2, "name": "Ward 02 - Badshahpur Highway", "village": "Badshahpur Rural Village", "zone": "Zone 4", "lat": 28.3980, "lon": 77.0540, "type": "Peri-Urban Rapid Expansion"},
                    {"id": 3, "name": "Ward 03 - Old Gurugram Sadar", "village": "Sadar Bazaar Mohalla", "zone": "Zone 1", "lat": 28.4680, "lon": 77.0320, "type": "Extreme Density Commercial Core"},
                    {"id": 4, "name": "Ward 04 - Sector 56 Golf Course Ext", "village": "Ghata Village Boundary", "zone": "Zone 3", "lat": 28.4230, "lon": 77.1080, "type": "Planned High-Rise Foothill"}
                ]
            }
        }
    },
    "Himachal Pradesh": {
        "districts": {
            "Shimla (SMC)": {
                "full_name": "Shimla Municipal Corporation (SMC)",
                "center_lat": 31.1048,
                "center_lon": 77.1734,
                "wards": [
                    {"id": 1, "name": "Ward 01 - The Ridge & Mall", "village": "Mall Heritage Locality", "zone": "Central", "lat": 31.1050, "lon": 77.1750, "type": "Pedestrian Heritage Core"},
                    {"id": 2, "name": "Ward 02 - Sanjauli Valley Ridge", "village": "Sanjauli Village Basti", "zone": "East", "lat": 31.1010, "lon": 77.1950, "type": "Steep Hillside High Density Risk"},
                    {"id": 3, "name": "Ward 03 - Chotta Shimla Govt Core", "village": "Kasumpti Foothill Village", "zone": "South", "lat": 31.0880, "lon": 31.0880, "type": "Administrative Mountain Slopes"},
                    {"id": 4, "name": "Ward 04 - Totu Railway Outskirts", "village": "Totu Gaon", "zone": "West", "lat": 31.1120, "lon": 77.1280, "type": "Toy Train Transit Valley"}
                ]
            }
        }
    },
    "Jharkhand": {
        "districts": {
            "Ranchi (RMC)": {
                "full_name": "Ranchi Municipal Corporation (RMC)",
                "center_lat": 23.3441,
                "center_lon": 85.3096,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Morabadi Ground", "village": "Morabadi Tribal Village", "zone": "Zone 1", "lat": 23.3890, "lon": 85.3280, "type": "Open Institutional & Sports Hub"},
                    {"id": 2, "name": "Ward 02 - Doranda Cantt", "village": "Doranda Old Basti", "zone": "Zone 4", "lat": 23.3320, "lon": 85.3210, "type": "Heritage Cantonment Dense Sector"},
                    {"id": 3, "name": "Ward 03 - Harmu Housing Colony", "village": "Harmu River Village", "zone": "Zone 3", "lat": 23.3510, "lon": 85.2980, "type": "Planned Residential Silt Stress"},
                    {"id": 4, "name": "Ward 04 - Kanke Dam Buffer", "village": "Kanke Agri Village", "zone": "Zone 1", "lat": 23.4210, "lon": 85.3180, "type": "Potable Water Reservoir Buffer"}
                ]
            }
        }
    },
    "Karnataka": {
        "districts": {
            "Bengaluru Urban (BBMP)": {
                "full_name": "Bruhat Bengaluru Mahanagara Palike (BBMP)",
                "center_lat": 12.9716,
                "center_lon": 77.5946,
                "wards": [
                    {"id": 1, "name": "Ward 80 - Koramangala Hub", "village": "Koramangala Village", "zone": "South Zone", "lat": 12.9352, "lon": 77.6245, "type": "Commercial & Tech Startup Hub"},
                    {"id": 2, "name": "Ward 110 - Indiranagar", "village": "Binnamangala Village", "zone": "East Zone", "lat": 12.9784, "lon": 77.6408, "type": "High Footfall Commercial"},
                    {"id": 3, "name": "Ward 149 - HSR Layout Sector 1-7", "village": "Agara Lake Village", "zone": "South Zone", "lat": 12.9121, "lon": 77.6446, "type": "Planned Tech Residential Sector"},
                    {"id": 4, "name": "Ward 150 - Bellandur Lake Catchment", "village": "Bellandur Revenue Village", "zone": "Mahadevapura", "lat": 12.9260, "lon": 77.6762, "type": "Severe Water Silt & Foam Deficit"},
                    {"id": 5, "name": "Ward 174 - Whitefield Tech Corridor", "village": "Kadugodi Village", "zone": "Mahadevapura", "lat": 12.9698, "lon": 77.7500, "type": "Heavy Freight & Pothole Stress"},
                    {"id": 6, "name": "Ward 25 - Peenya Industrial Estate", "village": "Peenya Village Basti", "zone": "Dasarahalli", "lat": 13.0285, "lon": 77.5197, "type": "Industrial Machinery Heavy Freight"}
                ]
            },
            "Mysuru (MCC)": {
                "full_name": "Mysuru City Corporation (MCC)",
                "center_lat": 12.2958,
                "center_lon": 76.6394,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Palace Heritage Core", "village": "Agrahara Mohalla", "zone": "Central", "lat": 12.3051, "lon": 76.6551, "type": "Heritage Conservation & High Tourism"},
                    {"id": 2, "name": "Ward 02 - Jayalakshmipuram", "village": "Jayalakshmipuram Area", "zone": "West", "lat": 12.3210, "lon": 76.6320, "type": "Established High Income Sector"},
                    {"id": 3, "name": "Ward 03 - Hebbal Industrial", "village": "Hebbal Lake Village", "zone": "North-West", "lat": 12.3610, "lon": 76.6020, "type": "Heavy Electronic Manufacturing"}
                ]
            }
        }
    },
    "Kerala": {
        "districts": {
            "Thiruvananthapuram (TMC)": {
                "full_name": "Thiruvananthapuram Municipal Corporation (TMC)",
                "center_lat": 8.5241,
                "center_lon": 76.9366,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Kowdiar Palace Area", "village": "Kowdiar Heritage Locality", "zone": "Central", "lat": 8.5280, "lon": 76.9610, "type": "Low Density Planned Green"},
                    {"id": 2, "name": "Ward 02 - Kazhakoottam Technopark", "village": "Kazhakoottam Coastal Village", "zone": "North", "lat": 8.5580, "lon": 76.8820, "type": "IT Tech Hub Rapid Influx"},
                    {"id": 3, "name": "Ward 03 - Vizhinjam Port Terminal", "village": "Vizhinjam Fishing Village", "zone": "South", "lat": 8.3780, "lon": 76.9920, "type": "Deepwater Port Coastal Logistics"},
                    {"id": 4, "name": "Ward 04 - Fort Padmanabhaswamy", "village": "East Fort Mohalla", "zone": "Central", "lat": 8.4830, "lon": 76.9440, "type": "Ancient Temple High Density Core"}
                ]
            },
            "Kochi (KMC)": {
                "full_name": "Kochi Municipal Corporation (KMC)",
                "center_lat": 9.9312,
                "center_lon": 76.2673,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Fort Kochi Heritage", "village": "Fort Kochi Coastal Village", "zone": "West", "lat": 9.9650, "lon": 76.2420, "type": "Colonial Coastal Tourism Core"},
                    {"id": 2, "name": "Ward 02 - Marine Drive Ernakulam", "village": "Broadway Wholesale Market", "zone": "Central", "lat": 9.9810, "lon": 76.2760, "type": "High Commercial Waterfront Core"},
                    {"id": 3, "name": "Ward 03 - Kakkanad InfoPark", "village": "Thrikkakara Gaon", "zone": "East", "lat": 10.0120, "lon": 76.3620, "type": "IT Tech Park & Heavy Traffic Grid"},
                    {"id": 4, "name": "Ward 04 - Edappally Metro Transit", "village": "Edappally Village Basti", "zone": "North", "lat": 10.0240, "lon": 76.3080, "type": "Multi-Modal Metro Interchange"}
                ]
            }
        }
    },
    "Madhya Pradesh": {
        "districts": {
            "Indore (IMC - Smart City)": {
                "full_name": "Indore Municipal Corporation (IMC - National Clean City Leader)",
                "center_lat": 22.7196,
                "center_lon": 75.8577,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Rajwada Heritage Core", "village": "Rajwada Old City Mohalla", "zone": "Central Zone", "lat": 22.7196, "lon": 75.8577, "type": "Heritage High Density Bazaar"},
                    {"id": 2, "name": "Ward 02 - Vijay Nagar Commercial", "village": "Bhamori Village Locality", "zone": "East Zone", "lat": 22.7533, "lon": 75.8937, "type": "High Footfall IT & Mall Corridor"},
                    {"id": 3, "name": "Ward 03 - Chhappan Dukan Food Street", "village": "Palasia Mohalla", "zone": "Central Zone", "lat": 22.7244, "lon": 75.8770, "type": "Zero-Waste Pedestrian Food Street"},
                    {"id": 4, "name": "Ward 04 - Sanwer Road Industrial", "village": "Sanwer Village Boundary", "zone": "North Zone", "lat": 22.7830, "lon": 75.8450, "type": "Sector A-E Heavy Freight Hub"},
                    {"id": 5, "name": "Ward 05 - Rau & Super Corridor", "village": "Rau Agricultural Village", "zone": "South-West Zone", "lat": 22.6390, "lon": 75.8050, "type": "Rapid Expansion Tech City Hub"}
                ]
            },
            "Bhopal (BMC)": {
                "full_name": "Bhopal Municipal Corporation (BMC)",
                "center_lat": 23.2599,
                "center_lon": 77.4126,
                "wards": [
                    {"id": 1, "name": "Ward 01 - VIP Road Upper Lake", "village": "Koh-e-Fiza Lake Village", "zone": "Lake Front", "lat": 23.2540, "lon": 77.3820, "type": "Scenic Lake Catchment Buffer"},
                    {"id": 2, "name": "Ward 02 - MP Nagar Commercial", "village": "Maharana Pratap Nagar", "zone": "Central", "lat": 23.2330, "lon": 77.4360, "type": "Central Business District"},
                    {"id": 3, "name": "Ward 03 - Kolar Road Suburban", "village": "Kolar Gaon", "zone": "South", "lat": 23.1670, "lon": 77.4210, "type": "Rapid Urbanizing Dense Grid"},
                    {"id": 4, "name": "Ward 04 - Govindpura Industrial", "village": "Govindpura BHEL Village", "zone": "East", "lat": 23.2610, "lon": 77.4640, "type": "Manufacturing Heavy Freight"}
                ]
            }
        }
    },
    "Maharashtra": {
        "districts": {
            "Mumbai City (BMC)": {
                "full_name": "Brihanmumbai Municipal Corporation (BMC)",
                "center_lat": 19.0760,
                "center_lon": 72.8777,
                "wards": [
                    {"id": 1, "name": "Ward A - Colaba & Fort", "village": "Colaba Koliwada Village", "zone": "South Mumbai", "lat": 18.9220, "lon": 72.8347, "type": "Heritage Coastal Commercial"},
                    {"id": 2, "name": "Ward D - Malabar Hill", "village": "Walkeshwar Temple Village", "zone": "South Mumbai", "lat": 18.9548, "lon": 72.8028, "type": "High Income Low Density"},
                    {"id": 3, "name": "Ward G/N - Dharavi Hub", "village": "Kumbharwada Artisan Village", "zone": "Central", "lat": 19.0434, "lon": 72.8567, "type": "Extreme Density Informal Economy"},
                    {"id": 4, "name": "Ward H/W - Bandra West", "village": "Ranwar Heritage Village", "zone": "Western Suburbs", "lat": 19.0596, "lon": 72.8295, "type": "Waterfront Commercial Corridor"},
                    {"id": 5, "name": "Ward H/E - Kurla West Basin", "village": "Kurla Gaon", "zone": "Eastern Suburbs", "lat": 19.0680, "lon": 72.8800, "type": "Mithi River Extreme Flood Basin"},
                    {"id": 6, "name": "Ward M/E - Govandi Deonar", "village": "Deonar Abattoir Village", "zone": "Eastern Suburbs", "lat": 19.0558, "lon": 72.9150, "type": "Dumping Ground High Health Risk"}
                ]
            },
            "Pune (PMC)": {
                "full_name": "Pune Municipal Corporation (PMC)",
                "center_lat": 18.5204,
                "center_lon": 73.8567,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Shivajinagar", "village": "Shivajinagar Gaothan", "zone": "Central", "lat": 18.5314, "lon": 73.8446, "type": "Administrative / Institutional"},
                    {"id": 2, "name": "Ward 02 - Kothrud Sector", "village": "Kothrud Gaon", "zone": "West", "lat": 18.5074, "lon": 73.8077, "type": "Dense Residential Sector"},
                    {"id": 3, "name": "Ward 03 - Viman Nagar IT Hub", "village": "Viman Nagar Locality", "zone": "East", "lat": 18.5679, "lon": 73.9143, "type": "Commercial & IT Corridor"},
                    {"id": 4, "name": "Ward 04 - Hinjewadi Corridor", "village": "Maan-Hinjewadi Village", "zone": "West", "lat": 18.5913, "lon": 73.7389, "type": "Major IT Tech Park Heavy Traffic"}
                ]
            },
            "Nagpur (NMC)": {
                "full_name": "Nagpur Municipal Corporation (NMC)",
                "center_lat": 21.1458,
                "center_lon": 79.0882,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Sitabuldi Central", "village": "Sitabuldi Fort Mohalla", "zone": "Central Zone", "lat": 21.1460, "lon": 79.0840, "type": "Transit Interchange & Wholesale"},
                    {"id": 2, "name": "Ward 02 - Dharampeth", "village": "Dharampeth Locality", "zone": "West Zone", "lat": 21.1480, "lon": 79.0620, "type": "Planned High Income Sector"},
                    {"id": 3, "name": "Ward 03 - Lakadganj Industrial", "village": "Lakadganj Timber Basti", "zone": "East Zone", "lat": 21.1520, "lon": 79.1240, "type": "Timber Woodwork & Effluent Silt"}
                ]
            }
        }
    },
    "Manipur": {
        "districts": {
            "Imphal (IMC)": {
                "full_name": "Imphal Municipal Corporation (IMC)",
                "center_lat": 24.8170,
                "center_lon": 93.9368,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Kangla Fort Core", "village": "Kangla Palace Heritage Village", "zone": "Central", "lat": 24.8190, "lon": 93.9410, "type": "Historic Palace Moat Zone"},
                    {"id": 2, "name": "Ward 02 - Thangal Bazaar", "village": "Ima Keithel Mothers Market", "zone": "Central", "lat": 24.8090, "lon": 93.9350, "type": "Historic Women-run Market Choke"},
                    {"id": 3, "name": "Ward 03 - Singjamei Riverbed", "village": "Singjamei Gaon", "zone": "South", "lat": 24.7820, "lon": 93.9380, "type": "Imphal River Flood Buffer"},
                    {"id": 4, "name": "Ward 04 - Lamphelpat Wetland", "village": "Lamphelpat Lowland Village", "zone": "West", "lat": 24.8340, "lon": 93.9150, "type": "Lowland Inundation Catchment"}
                ]
            }
        }
    },
    "Meghalaya": {
        "districts": {
            "Shillong (SMB)": {
                "full_name": "Shillong Municipal Board (SMB)",
                "center_lat": 25.5788,
                "center_lon": 91.8933,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Police Bazaar Central", "village": "Police Bazaar Shyiap", "zone": "Central", "lat": 25.5780, "lon": 91.8840, "type": "Hilly Commercial Interchange"},
                    {"id": 2, "name": "Ward 02 - Laitumkhrah Hill", "village": "Laitumkhrah Dorbar Village", "zone": "East", "lat": 25.5690, "lon": 91.9020, "type": "Institutional & Steep Gradient"},
                    {"id": 3, "name": "Ward 03 - Nongthymmai Suburban", "village": "Nongthymmai Khasi Village", "zone": "South-East", "lat": 25.5560, "lon": 91.9140, "type": "Dense Pine Ridge Settlement"},
                    {"id": 4, "name": "Ward 04 - Polo Ground Valley", "village": "Wahingdoh River Stream", "zone": "North", "lat": 25.5890, "lon": 91.8960, "type": "Valley Basin Inundation Risk"}
                ]
            }
        }
    },
    "Mizoram": {
        "districts": {
            "Aizawl (AMC)": {
                "full_name": "Aizawl Municipal Corporation (AMC)",
                "center_lat": 23.7271,
                "center_lon": 92.7176,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Zarkawt Central", "village": "Zarkawt Ridge Locality", "zone": "Central", "lat": 23.7340, "lon": 92.7210, "type": "Ridge Top Highway Bazaar"},
                    {"id": 2, "name": "Ward 02 - Khatla South", "village": "Khatla Tribal Basti", "zone": "South", "lat": 23.7180, "lon": 92.7120, "type": "Steep Slope Landslide Prone"},
                    {"id": 3, "name": "Ward 03 - Bawngkawn Transit", "village": "Bawngkawn Valley Village", "zone": "North", "lat": 23.7620, "lon": 92.7350, "type": "Northern Gateway Cargo Junction"}
                ]
            }
        }
    },
    "Nagaland": {
        "districts": {
            "Kohima (KMC)": {
                "full_name": "Kohima Municipal Council (KMC)",
                "center_lat": 25.6751,
                "center_lon": 94.1086,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Razhu Point Bazaar", "village": "Kohima Village Basti", "zone": "Central", "lat": 25.6740, "lon": 94.1090, "type": "Mountain Ridge Commercial Core"},
                    {"id": 2, "name": "Ward 02 - High School Hill", "village": "Phezoucha Gaon", "zone": "North", "lat": 25.6980, "lon": 94.1020, "type": "Steep Terraced Institutional"},
                    {"id": 3, "name": "Ward 03 - Ministers Hill VIP", "village": "Ministers Hill Locality", "zone": "South", "lat": 25.6580, "lon": 94.1140, "type": "Planned Low Density Slopes"}
                ]
            }
        }
    },
    "Odisha": {
        "districts": {
            "Bhubaneswar (BMC)": {
                "full_name": "Bhubaneswar Municipal Corporation (BMC)",
                "center_lat": 20.2961,
                "center_lon": 85.8245,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Saheed Nagar Central", "village": "Saheed Nagar Locality", "zone": "Central Zone", "lat": 20.2980, "lon": 85.8450, "type": "Planned Commercial Sector"},
                    {"id": 2, "name": "Ward 02 - Patia Infocity Hub", "village": "Patia Agricultural Village", "zone": "North Zone", "lat": 20.3540, "lon": 85.8180, "type": "Global IT Tech City Grid"},
                    {"id": 3, "name": "Ward 03 - Khandagiri Caves Ward", "village": "Khandagiri Heritage Village", "zone": "South-West", "lat": 20.2580, "lon": 85.7820, "type": "Monolithic Rock Heritage Tourism"},
                    {"id": 4, "name": "Ward 04 - Old Town Lingaraj Temple", "village": "Ekamra Kshetra Mohalla", "zone": "South Zone", "lat": 20.2410, "lon": 85.8340, "type": "Ancient Temple High Density Core"}
                ]
            }
        }
    },
    "Punjab": {
        "districts": {
            "Amritsar (AMC)": {
                "full_name": "Amritsar Municipal Corporation (AMC)",
                "center_lat": 31.6340,
                "center_lon": 74.8723,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Golden Temple Heritage", "village": "Katra Ahluwalia Mohalla", "zone": "Walled City", "lat": 31.6200, "lon": 74.8765, "type": "Pedestrian Heritage Religious Core"},
                    {"id": 2, "name": "Ward 02 - Ranjit Avenue Planned", "village": "Gumtala Gaon Boundary", "zone": "North Zone", "lat": 31.6580, "lon": 74.8540, "type": "Planned Commercial Sector"},
                    {"id": 3, "name": "Ward 03 - Chheharta Industrial", "village": "Chheharta Rural Village", "zone": "West Zone", "lat": 31.6310, "lon": 74.7890, "type": "Railway Siding Textile Mill Zone"},
                    {"id": 4, "name": "Ward 04 - Putligarh Dense Corridor", "village": "Putligarh Basti", "zone": "Central Zone", "lat": 31.6360, "lon": 74.8420, "type": "High Footfall Congested Market"}
                ]
            }
        }
    },
    "Rajasthan": {
        "districts": {
            "Jaipur (JMC)": {
                "full_name": "Jaipur Municipal Corporation (JMC)",
                "center_lat": 26.9124,
                "center_lon": 75.7873,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Walled City Johari Bazaar", "village": "Johari Bazaar Chowk", "zone": "Heritage Zone", "lat": 26.9200, "lon": 75.8240, "type": "UNESCO Heritage Dense Market"},
                    {"id": 2, "name": "Ward 02 - C-Scheme Administrative", "village": "Civil Lines Area", "zone": "Central Zone", "lat": 26.9070, "lon": 75.8010, "type": "Government Offices Prime Commercial"},
                    {"id": 3, "name": "Ward 03 - Mansarovar Housing Sector", "village": "Mansarovar Gaon", "zone": "South Zone", "lat": 26.8620, "lon": 75.7640, "type": "Large Planned Residential Colony"},
                    {"id": 4, "name": "Ward 04 - Sitapura Industrial Area", "village": "Sitapura Rural Village", "zone": "South Zone", "lat": 26.7780, "lon": 75.8390, "type": "Jewellery Export Industrial Park"}
                ]
            }
        }
    },
    "Sikkim": {
        "districts": {
            "Gangtok (GMC)": {
                "full_name": "Gangtok Municipal Corporation (GMC)",
                "center_lat": 27.3389,
                "center_lon": 88.6065,
                "wards": [
                    {"id": 1, "name": "Ward 01 - MG Marg Pedestrian", "village": "MG Marg Ridge Area", "zone": "Central", "lat": 27.3310, "lon": 88.6130, "type": "Eco-Friendly Pedestrian Promenade"},
                    {"id": 2, "name": "Ward 02 - Deorali Cable Car Zone", "village": "Deorali Basti", "zone": "South", "lat": 27.3190, "lon": 88.6020, "type": "Transit Ropeway Mountain Slopes"},
                    {"id": 3, "name": "Ward 03 - Tadong University Slopes", "village": "Tadong Gaon", "zone": "South", "lat": 27.3080, "lon": 88.5940, "type": "High Youth Population Valley"}
                ]
            }
        }
    },
    "Tamil Nadu": {
        "districts": {
            "Chennai (GCC)": {
                "full_name": "Greater Chennai Corporation (GCC)",
                "center_lat": 13.0827,
                "center_lon": 80.2707,
                "wards": [
                    {"id": 1, "name": "Ward 112 - T. Nagar Retail Hub", "village": "Thyagaraya Nagar Locality", "zone": "Central", "lat": 13.0418, "lon": 80.2341, "type": "Commercial Pedestrian Plaza"},
                    {"id": 2, "name": "Ward 102 - Anna Nagar Planned", "village": "Naduvakkarai Village", "zone": "Central", "lat": 13.0850, "lon": 80.2100, "type": "Planned High Density Grid"},
                    {"id": 3, "name": "Ward 124 - Mylapore Heritage", "village": "Mylapore Fishery Village", "zone": "South", "lat": 13.0330, "lon": 80.2680, "type": "Cultural Heritage Coastal Proximity"},
                    {"id": 4, "name": "Ward 177 - Velachery Lowland", "village": "Velachery Marshland Village", "zone": "South", "lat": 12.9810, "lon": 80.2180, "type": "Extreme Monsoon Flood Inundation"},
                    {"id": 5, "name": "Ward 180 - OMR Tech Corridor", "village": "Thoraipakkam Village", "zone": "South", "lat": 12.9340, "lon": 80.2310, "type": "IT Expressway & Rapid Growth"}
                ]
            },
            "Coimbatore (CCMC)": {
                "full_name": "Coimbatore City Municipal Corporation (CCMC)",
                "center_lat": 11.0168,
                "center_lon": 76.9558,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Gandhipuram Central", "village": "Gandhipuram Market Locality", "zone": "Central", "lat": 11.0180, "lon": 76.9670, "type": "Transit Interchange & Retail Core"},
                    {"id": 2, "name": "Ward 02 - Peelamedu Tech Park", "village": "Peelamedu Revenue Village", "zone": "East", "lat": 11.0310, "lon": 77.0200, "type": "Institutions & IT Tech Hub"},
                    {"id": 3, "name": "Ward 03 - Singanallur Lake Catchment", "village": "Singanallur Village", "zone": "East", "lat": 10.9980, "lon": 77.0250, "type": "Water Catchment Silt Stress"}
                ]
            }
        }
    },
    "Telangana": {
        "districts": {
            "Hyderabad (GHMC)": {
                "full_name": "Greater Hyderabad Municipal Corporation (GHMC)",
                "center_lat": 17.3850,
                "center_lon": 78.4867,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Charminar Old City", "village": "Moghulpura Heritage Mohalla", "zone": "South Zone", "lat": 17.3616, "lon": 78.4747, "type": "Historic High Density Traditional Core"},
                    {"id": 2, "name": "Ward 02 - Hitec City & Cyberabad", "village": "Madhapur Revenue Village", "zone": "West Zone", "lat": 17.4474, "lon": 78.3762, "type": "Global IT Tech Hub Heavy Influx"},
                    {"id": 3, "name": "Ward 03 - Banjara Hills Elite", "village": "Shaikpet Village Boundary", "zone": "Central Zone", "lat": 17.4156, "lon": 78.4350, "type": "High Income Undulating Hill Slopes"},
                    {"id": 4, "name": "Ward 04 - Secunderabad Station", "village": "Secunderabad Cantonment Area", "zone": "North Zone", "lat": 17.4399, "lon": 78.5017, "type": "Heavy Railway Transit Choke Point"},
                    {"id": 5, "name": "Ward 05 - Kukatpally Housing Board", "village": "Kukatpally Village", "zone": "North-West Zone", "lat": 17.4933, "lon": 78.3995, "type": "High Density Suburban Grid"}
                ]
            }
        }
    },
    "Tripura": {
        "districts": {
            "Agartala (AMC)": {
                "full_name": "Agartala Municipal Corporation (AMC)",
                "center_lat": 23.8315,
                "center_lon": 91.2868,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Ujjayanta Palace Core", "village": "Palace Compound Mohalla", "zone": "Central", "lat": 23.8360, "lon": 91.2840, "type": "Royal Heritage Moat Buffer"},
                    {"id": 2, "name": "Ward 02 - Battala Wholesale", "village": "Battala River Village", "zone": "South", "lat": 23.8240, "lon": 91.2780, "type": "Howrah River Inundation Area"},
                    {"id": 3, "name": "Ward 03 - Badharghat Outer", "village": "Badharghat Gaon", "zone": "South-West", "lat": 23.8050, "lon": 91.2690, "type": "Railway Terminal Freight Corridor"}
                ]
            }
        }
    },
    "Uttar Pradesh": {
        "districts": {
            "Lucknow (LMC)": {
                "full_name": "Lucknow Municipal Corporation (LMC)",
                "center_lat": 26.8467,
                "center_lon": 80.9462,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Hazratganj Prime Core", "village": "Narhi Old Mohalla", "zone": "Zone 1", "lat": 26.8467, "lon": 80.9462, "type": "Prime Commercial Corridor"},
                    {"id": 2, "name": "Ward 02 - Gomti Nagar Extension", "village": "Ujariyaon Gaon", "zone": "Zone 4", "lat": 26.8530, "lon": 81.0010, "type": "Modern Suburban Planned Sector"},
                    {"id": 3, "name": "Ward 03 - Chowk Heritage Basti", "village": "Chowk Old Walled Mohalla", "zone": "Zone 6", "lat": 26.8687, "lon": 80.9080, "type": "Extreme Density Traditional Basti"},
                    {"id": 4, "name": "Ward 04 - Aminabad Bazaar Choke", "village": "Aminabad Market Area", "zone": "Zone 1", "lat": 26.8436, "lon": 80.9256, "type": "Wholesale Retail Heavy Footfall"},
                    {"id": 5, "name": "Ward 05 - Alambagh Railway Hub", "village": "Alambagh Purana Gaon", "zone": "Zone 5", "lat": 26.8080, "lon": 80.8980, "type": "Heavy Transit Railway Corridor"}
                ]
            },
            "Varanasi (VNN)": {
                "full_name": "Varanasi Nagar Nigam (VNN)",
                "center_lat": 25.3176,
                "center_lon": 82.9739,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Dashashwamedh Ghat", "village": "Dashashwamedh Riverfront Basti", "zone": "Ghat Zone", "lat": 25.3075, "lon": 83.0105, "type": "Religious Heritage Ganga Corridor"},
                    {"id": 2, "name": "Ward 02 - Godowlia Bazaar Core", "village": "Godowlia Chowk Mohalla", "zone": "Core Zone", "lat": 25.3110, "lon": 83.0040, "type": "Extreme Pedestrian Footfall"},
                    {"id": 3, "name": "Ward 03 - Assi Ghat University", "village": "Assi Village Buffer", "zone": "South Zone", "lat": 25.2890, "lon": 83.0060, "type": "Cultural Heritage & Academic Hub"},
                    {"id": 4, "name": "Ward 04 - Shivpur Outer Corridor", "village": "Shivpur Agricultural Village", "zone": "North Zone", "lat": 25.3620, "lon": 82.9630, "type": "Highway Freight Expanding Grid"}
                ]
            },
            "Kanpur (KNN)": {
                "full_name": "Kanpur Nagar Nigam (KNN)",
                "center_lat": 26.4499,
                "center_lon": 80.3319,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Civil Lines", "village": "Civil Lines VIP Area", "zone": "Central Zone", "lat": 26.4710, "lon": 80.3420, "type": "Administrative Offices Hub"},
                    {"id": 2, "name": "Ward 02 - Jajmau Leather Cluster", "village": "Jajmau Industrial Village", "zone": "East Zone", "lat": 26.4320, "lon": 80.4010, "type": "Tannery Effluent Silt Zone"},
                    {"id": 3, "name": "Ward 03 - Kalyanpur IIT Tech", "village": "Kalyanpur Gaon", "zone": "West Zone", "lat": 26.5020, "lon": 80.2450, "type": "Institutional & Metro Corridor"}
                ]
            }
        }
    },
    "Uttarakhand": {
        "districts": {
            "Dehradun (DMC)": {
                "full_name": "Dehradun Municipal Corporation (DMC)",
                "center_lat": 30.3165,
                "center_lon": 78.0322,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Clock Tower Paltan Bazaar", "village": "Paltan Bazaar Mohalla", "zone": "Central", "lat": 30.3240, "lon": 78.0420, "type": "High Density Commercial Core"},
                    {"id": 2, "name": "Ward 02 - Rajpur Road Foothill", "village": "Rajpur Foothill Village", "zone": "North", "lat": 30.3680, "lon": 78.0820, "type": "Mussoorie Foothill Slopes"},
                    {"id": 3, "name": "Ward 03 - Clement Town Cantt", "village": "Clement Town Tibetan Village", "zone": "South", "lat": 30.2640, "lon": 78.0050, "type": "Educational & Low Density"}
                ]
            }
        }
    },
    "West Bengal": {
        "districts": {
            "Kolkata (KMC)": {
                "full_name": "Kolkata Municipal Corporation (KMC)",
                "center_lat": 22.5726,
                "center_lon": 88.3639,
                "wards": [
                    {"id": 1, "name": "Ward 063 - Park Street Heritage", "village": "Park Street Locality", "zone": "Central Zone", "lat": 22.5515, "lon": 88.3524, "type": "Colonial Heritage Commercial"},
                    {"id": 2, "name": "Ward 042 - Burrabazar Wholesale", "village": "Burrabazar Old Mandi Basti", "zone": "Central Zone", "lat": 22.5830, "lon": 88.3560, "type": "Extreme Wholesale Congestion"},
                    {"id": 3, "name": "Ward 085 - Ballygunge Elite", "village": "Ballygunge Circular Area", "zone": "South Zone", "lat": 22.5280, "lon": 88.3650, "type": "Planned High Income Sector"},
                    {"id": 4, "name": "Ward 108 - Salt Lake Wetland", "village": "Dhapa East Wetland Village", "zone": "East Zone", "lat": 22.5480, "lon": 88.4040, "type": "Waterlogged Waste Treatment"}
                ]
            }
        }
    },
    "Delhi NCT": {
        "districts": {
            "Central & North Delhi (MCD)": {
                "full_name": "Municipal Corporation of Delhi - North & Central Zone",
                "center_lat": 28.6500,
                "center_lon": 77.2200,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Civil Lines", "village": "Civil Lines Revenue Village", "zone": "North", "lat": 28.6812, "lon": 77.2227, "type": "Administrative / Heritage"},
                    {"id": 2, "name": "Ward 02 - Chandni Chowk", "village": "Chandni Chowk Katra Mohalla", "zone": "Central", "lat": 28.6506, "lon": 77.2303, "type": "High-Density Commercial"},
                    {"id": 3, "name": "Ward 03 - Connaught Place", "village": "CP Inner Circle Locality", "zone": "Central", "lat": 28.6328, "lon": 77.2197, "type": "Prime Commercial"},
                    {"id": 4, "name": "Ward 04 - Karol Bagh", "village": "Karol Bagh Market Basti", "zone": "Central", "lat": 28.6514, "lon": 77.1907, "type": "Mixed Commercial / Residential"},
                    {"id": 5, "name": "Ward 05 - Kashmere Gate", "village": "Kashmere Gate ISBT Village", "zone": "North", "lat": 28.6665, "lon": 77.2285, "type": "Transit & Heritage Hub"},
                    {"id": 6, "name": "Ward 06 - Model Town", "village": "Model Town Planned Village", "zone": "North", "lat": 28.7027, "lon": 77.1932, "type": "Established Residential"}
                ]
            },
            "South & East Delhi (MCD)": {
                "full_name": "Municipal Corporation of Delhi - South & East Zone",
                "center_lat": 28.5800,
                "center_lon": 77.2600,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Okhla Phase-II", "village": "Okhla Industrial Village", "zone": "South", "lat": 28.5308, "lon": 77.2711, "type": "Industrial Heavy Freight"},
                    {"id": 2, "name": "Ward 02 - Mayur Vihar", "village": "Chilla Village Boundary", "zone": "East", "lat": 28.6080, "lon": 77.2965, "type": "Planned Residential Sector"},
                    {"id": 3, "name": "Ward 03 - Shahdara North", "village": "Shahdara Old Mohalla", "zone": "East", "lat": 28.6734, "lon": 77.2907, "type": "High-Density Congested Core"},
                    {"id": 4, "name": "Ward 04 - Lajpat Nagar", "village": "Lajpat Nagar Central Market", "zone": "South", "lat": 28.5700, "lon": 77.2400, "type": "High Footfall Commercial"}
                ]
            },
            "West & South-West Delhi (MCD)": {
                "full_name": "Municipal Corporation of Delhi - West Zone",
                "center_lat": 28.6300,
                "center_lon": 77.0800,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Janakpuri", "village": "Possangipur Village", "zone": "West", "lat": 28.6219, "lon": 77.0878, "type": "Established Residential"},
                    {"id": 2, "name": "Ward 02 - Dwarka Sector-10", "village": "Bharthal Rural Village", "zone": "South-West", "lat": 28.5823, "lon": 77.0500, "type": "Sub-City Planned Sector"},
                    {"id": 3, "name": "Ward 03 - Najafgarh Outer", "village": "Najafgarh Dehat Village", "zone": "South-West", "lat": 28.6092, "lon": 76.9854, "type": "Peri-Urban Agricultural Influx"}
                ]
            }
        }
    },
    "Jammu & Kashmir": {
        "districts": {
            "Srinagar (SMC)": {
                "full_name": "Srinagar Municipal Corporation (SMC)",
                "center_lat": 34.0837,
                "center_lon": 74.7973,
                "wards": [
                    {"id": 1, "name": "Ward 01 - Lal Chowk Central", "village": "Maisuma Historic Mohalla", "zone": "Central", "lat": 34.0720, "lon": 74.8100, "type": "Commercial Heritage Core"},
                    {"id": 2, "name": "Ward 02 - Dal Lake Boulevard", "village": "Mir Behri Floating Water Village", "zone": "Lake Front", "lat": 34.0910, "lon": 74.8340, "type": "Ecological Lake Wetland Area"},
                    {"id": 3, "name": "Ward 03 - Hazratbal Shrine", "village": "Dargah Hazratbal Mohalla", "zone": "North", "lat": 34.1290, "lon": 74.8420, "type": "Pilgrim Heritage Waterfront"}
                ]
            }
        }
    }
}

STATE_ICONS = {
    "Andhra Pradesh": "🏛️",
    "Arunachal Pradesh": "🏔️",
    "Assam": "🍵",
    "Bihar": "🌾",
    "Chhattisgarh": "⛏️",
    "Goa": "🏖️",
    "Gujarat": "🏭",
    "Haryana": "🚜",
    "Himachal Pradesh": "❄️",
    "Jharkhand": "🌲",
    "Karnataka": "💻",
    "Kerala": "🥥",
    "Madhya Pradesh": "⭐",
    "Maharashtra": "🌊",
    "Manipur": "💎",
    "Meghalaya": "☁️",
    "Mizoram": "🎋",
    "Nagaland": "🦅",
    "Odisha": "🛕",
    "Punjab": "🌾",
    "Rajasthan": "🏰",
    "Sikkim": "🌸",
    "Tamil Nadu": "🌴",
    "Telangana": "🌆",
    "Tripura": "🎍",
    "Uttar Pradesh": "🕉️",
    "Uttarakhand": "⛰️",
    "West Bengal": "🌺",
    "Delhi NCT": "🏛️",
    "Jammu & Kashmir": "🏔️"
}

# Backward compatibility alias
CITY_METADATA = {
    f"{state} - {dist}": dist_info
    for state, state_data in STATE_DISTRICT_DATABASE.items()
    for dist, dist_info in state_data["districts"].items()
}

def generate_ward_dataset(state_name: str = "Delhi NCT", district_name: str = "Central & North Delhi (MCD)", seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic municipal dataset for the selected State and District,
    including authentic village / mohalla locality names and precise GIS coordinates.
    """
    np.random.seed(seed)
    
    state_obj = STATE_DISTRICT_DATABASE.get(state_name, STATE_DISTRICT_DATABASE["Delhi NCT"])
    dist_map = state_obj["districts"]
    dist_obj = dist_map.get(district_name, list(dist_map.values())[0])
    wards_metadata = dist_obj["wards"]

    records = []

    for w in wards_metadata:
        is_dense = any(k in w["type"] for k in ["Dense", "Congested", "Choke", "Industrial", "Flood", "Pothole", "Ghat", "Extreme", "Basti"])
        is_posh = any(k in w["type"] for k in ["Planned", "Prime", "Heritage", "High Income", "High-Income", "High Street", "Riverfront", "VIP"])
        is_commercial = any(k in w["type"] for k in ["Commercial", "Bazaar", "Hub", "Retail", "Market"])

        population = int(np.random.uniform(120000, 270000) if is_dense else np.random.uniform(55000, 140000))
        area_sqkm = round(float(np.random.uniform(3.5, 12.0)), 2)
        pop_density = int(population / area_sqkm)

        avg_road_age_years = round(float(np.random.uniform(7.0, 14.5) if is_dense else np.random.uniform(1.5, 5.0) if is_posh else np.random.uniform(4.5, 9.0)), 1)
        water_deficit_pct = round(float(np.random.uniform(38.0, 72.0) if is_dense else np.random.uniform(5.0, 22.0) if is_posh else np.random.uniform(22.0, 44.0)), 1)
        drainage_choke_idx = round(float(np.random.uniform(60, 98) if is_dense else np.random.uniform(12, 35) if is_posh else np.random.uniform(38, 68)), 1)
        citizen_complaints_30d = int(np.random.uniform(180, 420) if is_dense else np.random.uniform(20, 85) if is_posh else np.random.uniform(90, 220))
        unresolved_potholes = int(np.random.uniform(40, 98) if is_dense else np.random.uniform(3, 14) if is_posh else np.random.uniform(16, 48))
        waste_frequency_wk = round(float(np.random.uniform(2.0, 3.8) if is_dense else np.random.uniform(5.8, 7.0) if is_posh else np.random.uniform(4.0, 5.5)), 1)

        nightlight_index = round(float(np.random.uniform(0.70, 0.98) if is_commercial or is_posh else np.random.uniform(0.32, 0.62)), 2)
        urban_heat_island_idx = round(float(np.random.uniform(3.5, 6.0) if is_dense or is_commercial else np.random.uniform(0.8, 2.2) if is_posh else np.random.uniform(2.0, 3.8)), 2)
        green_cover_pct = round(float(np.random.uniform(25.0, 46.0) if is_posh else np.random.uniform(3.0, 11.0) if is_dense else np.random.uniform(12.0, 25.0)), 1)
        iot_pipeline_pressure_psi = round(float(np.random.uniform(16.0, 26.0) if is_dense else np.random.uniform(48.0, 65.0) if is_posh else np.random.uniform(30.0, 46.0)), 1)
        iot_drain_silt_level_pct = round(float(np.random.uniform(65, 96) if is_dense else np.random.uniform(12, 32) if is_posh else np.random.uniform(38, 65)), 1)

        current_budget_lakhs = round(float(np.random.uniform(95.0, 240.0)), 1)

        norm_road_age = (avg_road_age_years / 14.5) * 100
        norm_complaints = min(100.0, (citizen_complaints_30d / 400.0) * 100)
        norm_potholes = min(100.0, (unresolved_potholes / 95.0) * 100)
        norm_waste_lag = ((7.0 - waste_frequency_wk) / 5.0) * 100
        norm_uhi = min(100.0, (urban_heat_island_idx / 6.0) * 100)
        norm_green_deficit = ((45.0 - green_cover_pct) / 45.0) * 100

        deficit_score = (
            0.22 * norm_road_age +
            0.24 * water_deficit_pct +
            0.18 * drainage_choke_idx +
            0.14 * norm_complaints +
            0.10 * norm_waste_lag +
            0.06 * norm_uhi +
            0.06 * norm_green_deficit
        )
        deficit_score = round(float(np.clip(deficit_score, 12.0, 96.5)), 1)

        if deficit_score >= 62.0:
            risk_category = "Critical"
            risk_color = "#EF4444"
        elif deficit_score >= 38.0:
            risk_category = "Moderate"
            risk_color = "#F59E0B"
        else:
            risk_category = "Stable"
            risk_color = "#10B981"

        village_label = w.get("village", w["name"].split(" - ")[-1] + " Locality")

        records.append({
            "ward_id": w["id"],
            "ward_name": w["name"],
            "village_name": village_label,
            "zone": w["zone"],
            "lat": w["lat"],
            "lon": w["lon"],
            "ward_type": w["type"],
            "population": population,
            "area_sqkm": area_sqkm,
            "pop_density": pop_density,
            "avg_road_age_years": avg_road_age_years,
            "water_deficit_pct": water_deficit_pct,
            "drainage_choke_idx": drainage_choke_idx,
            "citizen_complaints_30d": citizen_complaints_30d,
            "unresolved_potholes": unresolved_potholes,
            "waste_frequency_wk": waste_frequency_wk,
            "nightlight_index": nightlight_index,
            "urban_heat_island_idx": urban_heat_island_idx,
            "green_cover_pct": green_cover_pct,
            "iot_pipeline_pressure_psi": iot_pipeline_pressure_psi,
            "iot_drain_silt_level_pct": iot_drain_silt_level_pct,
            "current_budget_lakhs": current_budget_lakhs,
            "deficit_score": deficit_score,
            "risk_category": risk_category,
            "risk_color": risk_color
        })

    df = pd.DataFrame(records)
    if "village_name" not in df.columns:
        df["village_name"] = df["ward_name"].apply(lambda x: x.split(" - ")[-1] + " Village")

    # K-Means clustering
    cluster_features = df[["lat", "lon", "deficit_score", "pop_density", "water_deficit_pct"]].copy()
    norm_cf = (cluster_features - cluster_features.mean()) / (cluster_features.std() + 1e-5)
    kmeans = KMeans(n_clusters=min(3, len(df)), random_state=seed, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(norm_cf)
    
    cluster_names = {
        0: "Zone A - High Deficit Core",
        1: "Zone B - Developing Outskirts",
        2: "Zone C - Planned Moderate Buffer"
    }
    df["cluster_label"] = df["kmeans_cluster"].map(cluster_names).fillna("Zone A - High Deficit Core")

    return df


def get_sector_breakdown(ward_row: pd.Series) -> dict:
    """Computes estimated deficit breakdown by sector."""
    total = ward_row["deficit_score"]
    road_share = (ward_row["avg_road_age_years"] * 6 + ward_row["unresolved_potholes"] * 0.8)
    water_share = (ward_row["water_deficit_pct"] * 1.2 + ward_row["drainage_choke_idx"] * 0.9)
    sanitation_share = (ward_row["citizen_complaints_30d"] * 0.25 + (7 - ward_row["waste_frequency_wk"]) * 10)
    green_share = (45 - ward_row["green_cover_pct"]) + (ward_row["urban_heat_island_idx"] * 5)
    denom = road_share + water_share + sanitation_share + green_share
    
    return {
        "Roads & Pavements": round((road_share / denom) * total, 1),
        "Water Supply & Sewage": round((water_share / denom) * total, 1),
        "Sanitation & Solid Waste": round((sanitation_share / denom) * total, 1),
        "Greenery & Urban Cooling": round((green_share / denom) * total, 1),
    }
