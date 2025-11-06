# Practical-1: MetaMask Installation and Ether Spending Study

## Step 1: Install MetaMask

### For Desktop (Browser)

1. Open your preferred browser (Chrome, Firefox, Edge, or Brave).
2. Go to the official MetaMask website: [https://metamask.io](https://metamask.io)
3. Click **"Download"** and select your browser.
4. Add the extension to your browser.
5. Open MetaMask, and follow the setup:
   - **Create a new wallet** or **import an existing one**.
   - Set a **strong password**.
   - **Backup your secret recovery phrase** securely.

### For Mobile

1. Go to your App Store (iOS) or Google Play Store (Android).
2. Search for **MetaMask**.
3. Download and install the app.
4. Follow similar setup steps (create wallet, backup phrase).

## Step 2: Connect MetaMask to a Network

- By default, MetaMask uses the **Ethereum Mainnet**.
- To use test networks for learning, click the network dropdown and select a testnet like **Goerli Test Network**.
- Obtain free test Ether from a Goerli faucet: [https://goerlifaucet.com/](https://goerlifaucet.com/).

## Step 3: Understanding Ether Spending per Transaction

### Key Concepts
1. **Gas Limit:** Maximum units of gas you are willing to spend.
2. **Gas Price:** Price per gas unit in **Gwei** (1 ETH = 1,000,000,000 Gwei).
3. **Transaction Fee = Gas Limit × Gas Price**

## Step 4: Make a Test Transaction

1. Ensure your wallet has ETH (test ETH on testnets).
2. Click **Send** in MetaMask.
3. Enter the recipient address and amount.
4. Adjust **Gas Fee** if desired.
5. Confirm the transaction.

## Step 5: Observe Spending

- After sending, MetaMask displays the **Transaction Fee** in ETH.
- To see detailed info, check **Etherscan**:
  1. Copy the transaction hash from MetaMask.
  2. Go to [https://etherscan.io](https://etherscan.io) or testnet explorer like [https://goerli.etherscan.io](https://goerli.etherscan.io).
  3. Paste the hash to see gas used and ETH spent.

## Step 6: Experiment

- Send multiple transactions with different **gas prices**.
- Observe how fees change based on **network congestion**.
- Try interacting with **smart contracts** to see higher gas usage.

## Tips
- Always use a testnet first to avoid losing real ETH.
- MetaMask provides detailed transaction history and gas usage.

