# WhatsApp Bot Support

## Evaluation Summary

**Yes, WhatsApp supports bots** via two distinct pathways:

1. **Official route — WhatsApp Business Platform (Cloud API)**: Meta's
   production-grade, officially supported API for businesses and developers.
   Requires business verification and a dedicated phone number.

2. **Personal account route — WhatsApp Web protocol (current Bantu approach)**:
   Uses the WhatsApp Web protocol via the open-source
   [Baileys](https://github.com/WhiskeySockets/Baileys) library. Links an
   existing personal or business account as a "Linked Device" — the same
   mechanism used by WhatsApp Web and WhatsApp Desktop. No Meta approval
   required but subject to WhatsApp's Terms of Service.

---

## Official Route: WhatsApp Business Platform (Cloud API)

Meta provides the [WhatsApp Business Platform](https://business.whatsapp.com/)
for businesses that want to programmatically send and receive messages at scale.

### Prerequisites

- A Meta Business account at [business.facebook.com](https://business.facebook.com/)
- A phone number that is **not** already registered on WhatsApp
- Business verification (required to scale beyond the sandbox tier)

### Registration Steps

**Step 1: Create a Meta app**

- Go to [developers.facebook.com](https://developers.facebook.com/) → **My Apps
  → Create App**
- Choose **Business** app type
- Add the **WhatsApp** product to your app

**Step 2: Set up a WhatsApp Business Account (WABA)**

- In the app dashboard, navigate to **WhatsApp → Getting Started**
- Create or link an existing WhatsApp Business Account
- Add a phone number; Meta provides a free test number for development

**Step 3: Configure a webhook**

- In **WhatsApp → Configuration**, set a webhook URL (must be HTTPS)
- Subscribe to the `messages` field to receive inbound messages
- Verify the webhook with a token you provide

**Step 4: Obtain an access token**

- Use a **Permanent System User token** from Meta Business Manager for
  production deployments (temporary tokens expire after a few hours)
- Store the token securely in your environment

**Step 5: Business verification** (required to move beyond sandbox limits)

- Complete Meta Business verification at
  [business.facebook.com/settings](https://business.facebook.com/settings)
- Required to send messages to users who have not opted in, and to raise
  per-day messaging limits

### Sending and receiving messages

| Direction | Mechanism |
|-----------|-----------|
| Outbound | `POST https://graph.facebook.com/v{version}/{phone-number-id}/messages` |
| Inbound | Delivered to your HTTPS webhook via HTTP POST |

**Important**: The first message to a user must use a pre-approved **message
template**. Free-form replies are only allowed within 24 hours of the user's
last inbound message (the "customer service window").

### Limitations

- Message templates must be submitted to Meta for review (typically 1–3
  business days)
- Messaging is rate-limited by tier (Tier 1: 1 000 unique conversations/day,
  scaling up after verification)
- Business verification is required to scale beyond the sandbox
- Outbound-only use cases (no user reply) require explicit opt-in

---

## Personal Account Route: WhatsApp Web Protocol (Baileys)

Bantu currently uses the [Baileys](https://github.com/WhiskeySockets/Baileys)
open-source library, which implements the WhatsApp Web multi-device protocol.
This links a regular WhatsApp account (personal or WhatsApp Business app) as an
additional "Linked Device" — the same mechanism used by the official WhatsApp
Web browser client.

### How it works

- No Meta developer account is needed
- Any existing phone number registered on WhatsApp can be used
- A Node.js bridge process connects to WhatsApp Web and exposes a local
  WebSocket (`ws://localhost:3001` by default)
- The Python gateway connects to the bridge over that WebSocket
- Session credentials are stored locally (`~/.bantu/whatsapp-auth/` by default)
  and reused on subsequent starts without requiring another QR scan

### Registration / Setup

**Step 1: Start the bridge and scan the QR code**

```bash
nanobot channels login
```

This starts the Baileys bridge. A QR code is printed in the terminal. Open
WhatsApp on your phone:

- **iOS / Android**: Settings → Linked Devices → Link a Device → scan the QR
  code

After scanning, the session credentials are saved. The bridge reconnects
automatically on future starts without another scan.

**Step 2: Configure the channel**

Add to `~/.bantu/config.json`:

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Set to `true` to activate the channel |
| `allowFrom` | list | `[]` | Phone numbers (E.164 format) allowed to interact with the bot. An empty list permits anyone. |
| `bridgeUrl` | string | `ws://localhost:3001` | WebSocket URL of the Node.js bridge |
| `bridgeToken` | string | `""` | Optional shared secret for bridge authentication (recommended when the bridge is not on localhost) |

**Step 3: Run gateway and bridge**

```bash
# Terminal 1: bridge (keeps the WhatsApp session alive)
nanobot channels login

# Terminal 2: gateway (connects to the bridge and starts the agent)
nanobot gateway
```

### Limitations

- Requires an existing personal or WhatsApp Business account (phone + SIM)
- WhatsApp may restrict or ban accounts it detects as automated bots
- Not officially sanctioned by Meta; violates WhatsApp's Terms of Service for
  commercial or high-volume messaging
- Group message support requires the bot account to be a member of the group
- No message template system — all messages are sent as free-form text
- Voice message transcription is not yet available for the WhatsApp channel
- A single WhatsApp account can be linked as up to four Linked Devices; each
  `nanobot channels login` session consumes one slot

---

## Comparison

| Criterion | Cloud API (official) | Baileys (current) |
|-----------|---------------------|-------------------|
| Meta approval required | Yes | No |
| Dedicated phone number | Required | Uses existing account |
| Template approval | Required for outbound | Not needed |
| Messaging volume | Tiered (1 000 → unlimited with verification) | Subject to spam detection |
| Inbound webhook | HTTPS endpoint | Local WebSocket bridge |
| Group support | Limited | Supported |
| Officially supported by Meta | Yes | No (Terms of Service risk) |
| Setup time | Days (business verification) | Minutes (QR scan) |
| Business verification | Required to scale | Not applicable |

---

## Recommendation

For **personal or small-team use**, the current Baileys approach (QR-code scan)
is sufficient and quick to set up. The existing `nanobot channels login` command
handles the entire registration flow.

For **business or production deployments** where reliability, compliance, and
Meta support are required, the **WhatsApp Cloud API** is the correct path.
Bantu does not currently include a built-in Cloud API channel adapter; such an
adapter would be a separate implementation task and would replace the
Baileys-based bridge with direct HTTPS calls to Meta's Graph API and an HTTPS
webhook endpoint.
