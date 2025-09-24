import React, { useState, useEffect } from 'react';

// The frontend team MUST add this script tag to their main public/index.html file:
// <script src="https://cdn.ethers.io/lib/ethers-5.7.2.umd.min.js" type="application/javascript"></script>

// --- Configuration (To be filled in by the frontend team) ---
const contractAddress ="0xd9145CCE52D386f254917e481eB44e9943F39138"; // The address of your deployed contract

// --- Contract ABI (You must provide this to the frontend team) ---
// Paste the full ABI JSON array you copied from Remix IDE here.
const contractABI = [
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
];

function Reports() {
    const [userAddress, setUserAddress] = useState(null);
    const [reports, setReports] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const connectWallet = async () => {
        if (typeof window.ethereum === 'undefined') {
            return setError("Please install MetaMask to use this feature.");
        }
        try {
            // Use the ethers library loaded from the script tag
            const provider = new window.ethers.providers.Web3Provider(window.ethereum);
            await provider.send("eth_requestAccounts", []);
            const signer = provider.getSigner();
            setUserAddress(await signer.getAddress());
        } catch (err) {
            console.error(err);
            setError("Wallet connection failed. Please try again.");
        }
    };

    useEffect(() => {
        const fetchReports = async () => {
            if (!userAddress) return;
            setIsLoading(true);
            setError('');
            try {
                const provider = new window.ethers.providers.Web3Provider(window.ethereum);
                const contract = new window.ethers.Contract(contractAddress, contractABI, provider);
                const reportHashes = await contract.getUserReports(userAddress);
                setReports(reportHashes);
            } catch (err) {
                console.error(err);
                setError("Failed to fetch reports from the blockchain.");
            } finally {
                setIsLoading(false);
            }
        };
        fetchReports();
    }, [userAddress]);

    return (
        <div className="bg-gray-800 p-6 rounded-lg shadow-xl text-white">
            <h2 className="text-xl font-bold mb-4 text-teal-400">Your Medical Reports</h2>
            {!userAddress ? (
                <button 
                  onClick={connectWallet} 
                  className="w-full bg-teal-500 hover:bg-teal-600 font-bold py-2 px-4 rounded transition-colors duration-300"
                >
                    Connect Wallet to View Reports
                </button>
            ) : (
                <p className="text-center text-xs text-gray-400 mb-4">
                    Connected: <span className="font-mono">{`${userAddress.substring(0, 6)}...${userAddress.substring(userAddress.length - 4)}`}</span>
                </p>
            )}

            {error && <p className="text-red-500 text-center my-2">{error}</p>}

            <div className="space-y-2 mt-4">
                {isLoading ? <p className="text-center text-gray-400">Loading reports...</p> : (
                    reports.length > 0 ? (
                        reports.map((hash, index) => (
                            <a 
                              key={index} 
                              href={`https://ipfs.io/ipfs/${hash}`} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="block bg-gray-700 p-3 rounded hover:bg-gray-600 font-mono text-sm transition-colors duration-300"
                            >
                                Report #{index + 1}: <span className="text-teal-400 underline">{hash.substring(0, 20)}...</span>
                            </a>
                        ))
                    ) : ( userAddress && <p className="text-gray-500 text-center">No reports found for this address.</p> )
                )}
            </div>
        </div>
    );
}

export default Reports;