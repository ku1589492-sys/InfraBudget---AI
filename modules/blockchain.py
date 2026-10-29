"""
Blockchain Fund Tracking & Smart Contract Escrow Simulator.
Provides cryptographic block-hashing (SHA-256), immutable audit ledgers,
and milestone-based fund release verification to eliminate municipal fund leakage.
"""

import hashlib
import time
import json
from datetime import datetime, timedelta
import pandas as pd

def calculate_sha256(data: str) -> str:
    """Computes SHA-256 hash string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

class MunicipalBlock:
    def __init__(self, index: int, timestamp: str, ward_name: str, project_name: str, 
                 amount_lakhs: float, milestone_phase: str, verification_proof: str, prev_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.ward_name = ward_name
        self.project_name = project_name
        self.amount_lakhs = amount_lakhs
        self.milestone_phase = milestone_phase
        self.verification_proof = verification_proof
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "ward_name": self.ward_name,
            "project_name": self.project_name,
            "amount_lakhs": self.amount_lakhs,
            "milestone_phase": self.milestone_phase,
            "verification_proof": self.verification_proof,
            "prev_hash": self.prev_hash
        }, sort_keys=True)
        return calculate_sha256(block_string)

def generate_blockchain_ledger(df: pd.DataFrame) -> list:
    """
    Simulates an immutable chain of transaction blocks for recent municipal disbursements.
    """
    blocks = []
    
    # Genesis Block
    genesis_time = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    genesis = MunicipalBlock(
        index=0,
        timestamp=genesis_time,
        ward_name="City Central Treasury",
        project_name="MCMC Capital Fund Initialization 2026-27",
        amount_lakhs=3200.0,
        milestone_phase="Genesis Escrow Initialization",
        verification_proof="Council Resolution Gazetted #MCMC-2026-FIN-01",
        prev_hash="0" * 64
    )
    blocks.append(genesis)

    # Sample municipal infrastructure projects across wards
    sample_disbursements = [
        ("Ward 05 - Okhla Phase-II", "Heavy Drainage Desilting & Sump Automata", 68.5, "Phase 1: Geo-Survey & Tender Escrow", "LIDAR Spatial Mesh #9102-OKH", "VERIFIED"),
        ("Ward 05 - Okhla Phase-II", "Heavy Drainage Desilting & Sump Automata", 82.0, "Phase 2: IoT Flow Sensors & Material Baseline", "IoT Smart Grid Telemetry #FLOW-OKH-04", "VERIFIED"),
        ("Ward 08 - Seelampur", "Congested Corridor Road Milling & Paving", 54.0, "Phase 1: Geo-Survey & Tender Escrow", "Drone Elevation Survey #DRN-SLM-88", "VERIFIED"),
        ("Ward 07 - Shahdara North", "Water Booster Station & Pipe Overhaul", 75.0, "Phase 1: Ultrasonic Pipe Integrity Audit", "Non-Destructive Sonar Ping #NDT-SHD-12", "VERIFIED"),
        ("Ward 07 - Shahdara North", "Water Booster Station & Pipe Overhaul", 60.0, "Phase 2: Pump Installation & Pressure Baseline", "Smart Meter Handshake #PR-992-SHD", "VERIFIED"),
        ("Ward 02 - Chandni Chowk", "Heritage Drainage & Underground Cable Ducting", 92.0, "Phase 1: Heritage Conservation Review", "Archeological Clearance #ASI-DL-302", "VERIFIED"),
        ("Ward 04 - Karol Bagh", "Pothole Rapid Patching & Bitumen Resurfacing", 48.0, "Phase 1: Pothole Laser Scanning", "Mobile LiDAR Surface Scan #MLS-KB-74", "VERIFIED"),
        ("Ward 13 - Najafgarh Outer", "Rainwater Harvesting Catchment Pit Network", 45.0, "Phase 1: Hydrogeological Soil Permeability Test", "Borehole Hydro-Scan #GW-NJF-09", "VERIFIED")
    ]

    base_date = datetime.now() - timedelta(days=60)
    for i, item in enumerate(sample_disbursements, start=1):
        block_time = (base_date + timedelta(days=i * 6)).strftime("%Y-%m-%d %H:%M:%S")
        new_block = MunicipalBlock(
            index=i,
            timestamp=block_time,
            ward_name=item[0],
            project_name=item[1],
            amount_lakhs=item[2],
            milestone_phase=item[3],
            verification_proof=item[4],
            prev_hash=blocks[-1].hash
        )
        blocks.append(new_block)

    return blocks


def get_smart_contract_phases(ward_name: str) -> list:
    """
    Returns the real-time smart contract escrow state for a given ward's active flagship project.
    """
    return [
        {
            "phase": "Phase 1: Geo-Survey & Smart Tender Escrow",
            "share_pct": 25,
            "status": "COMPLETED & DISBURSED",
            "proof_type": "Satellite LiDAR & Drone Elevation Mesh",
            "oracle_contract": "0x7F2a...890C",
            "escrow_state": "Released (100%)",
            "color": "#10B981"
        },
        {
            "phase": "Phase 2: Material Procurement & IoT Baseline Sensor Setup",
            "share_pct": 30,
            "status": "COMPLETED & DISBURSED",
            "proof_type": "RFID Supply-Chain Stamp & IoT Water/Drain Flow Ping",
            "oracle_contract": "0x3B4c...11FA",
            "escrow_state": "Released (100%)",
            "color": "#10B981"
        },
        {
            "phase": "Phase 3: Civil Construction & Re-engineering Milestones",
            "share_pct": 30,
            "status": "IN-PROGRESS / ESCROW UNDER VERIFICATION",
            "proof_type": "Automated Drone Video Inspection & Bi-weekly Site Audit",
            "oracle_contract": "0x8E19...4D92",
            "escrow_state": "Partial Release (50% Locked)",
            "color": "#F59E0B"
        },
        {
            "phase": "Phase 4: Citizen Sign-Off & Public Satisfaction Quorum",
            "share_pct": 15,
            "status": "LOCKED IN ESCROW (ZERO-LEAKAGE SAFETY)",
            "proof_type": "Participatory Citizen Portal Voting & Grievance Resolution (>75%)",
            "oracle_contract": "0x5A02...E349",
            "escrow_state": "Fully Locked until Public Audit",
            "color": "#64748B"
        }
    ]
