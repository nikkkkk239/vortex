import os
from web3 import Web3
from dotenv import load_dotenv

# The backend team must create a .env file with these variables in their project's root.
load_dotenv()

# --- Configuration (To be filled in by the backend team) ---
NODE_PROVIDER_URL = os.getenv("NODE_PROVIDER_URL") # Your Infura/Alchemy URL for the Sepolia testnet
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")   # The address of your deployed contract
SERVICE_PRIVATE_KEY = os.getenv("SERVICE_PRIVATE_KEY") # The private key of the wallet you used to deploy
SERVICE_ADDRESS = os.getenv("SERVICE_ADDRESS")         # The public address of that same wallet

# --- Contract ABI (You must provide this to the backend team) ---
# Paste the full ABI JSON array you copied from Remix IDE here.
CONTRACT_ABI = """
[
  {
    "inputs": [],
    "stateMutability": "nonpayable",
    "type": "constructor"
  },
  {
    "anonymous": false,
    "inputs": [
      {
        "indexed": true,
        "internalType": "address",
        "name": "user",
        "type": "address"
      },
      {
        "indexed": false,
        "internalType": "string",
        "name": "reportHash",
        "type": "string"
      },
      {
        "indexed": false,
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      }
    ],
    "name": "ReportDelivered",
    "type": "event"
  },
  {
    "inputs": [],
    "name": "authorizedService",
    "outputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "userAddress",
        "type": "address"
      },
      {
        "internalType": "string",
        "name": "_reportHash",
        "type": "string"
      }
    ],
    "name": "deliverReport",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "userAddress",
        "type": "address"
      }
    ],
    "name": "getUserReports",
    "outputs": [
      {
        "internalType": "string[]",
        "name": "",
        "type": "string[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "userReports",
    "outputs": [
      {
        "internalType": "string",
        "name": "reportHash",
        "type": "string"
      },
      {
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  }
]
"""

def record_report_delivery(user_address: str, report_ipfs_hash: str) -> str:
    """
    Connects to the blockchain and calls the deliverReport function on the smart contract.

    Args:
        user_address (str): The patient's Ethereum wallet address.
        report_ipfs_hash (str): The IPFS hash (CID) of the generated report.

    Returns:
        str: The transaction hash as a hex string if successful, otherwise None.
    """
    try:
        w3 = Web3(Web3.HTTPProvider(NODE_PROVIDER_URL))
        if not w3.is_connected():
            raise ConnectionError("Failed to connect to the blockchain node.")
        
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
        nonce = w3.eth.get_transaction_count(SERVICE_ADDRESS)
        
        # Build the transaction
        tx = contract.functions.deliverReport(user_address, report_ipfs_hash).build_transaction({
            'chainId': 11155111, # Sepolia testnet chain ID. Change if using another network.
            'gas': 200000,
            'gasPrice': w3.to_wei('50', 'gwei'),
            'nonce': nonce,
        })
        
        # Sign and send the transaction
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=SERVICE_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for the transaction to be confirmed
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        print(f"Transaction successful! Hash: {tx_hash.hex()}")
        return tx_hash.hex()
        
    except Exception as e:
        print(f"An error occurred during the blockchain transaction: {e}")
        return None