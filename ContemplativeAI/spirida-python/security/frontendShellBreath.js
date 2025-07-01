// frontendShellBreath.js  (pseudocode – not yet wired to any framework)
import { sha256 } from "./tinySha.js";   // any 200-B hash util
const REST_WINDOW = 256;
let restTimes = [];

function onRestPhase() {
  restTimes.push(Date.now());
  if (restTimes.length > REST_WINDOW) restTimes.shift();
}

function currentDigest() {
  if (restTimes.length < 2) return "0".repeat(64);
  const deltas = restTimes.slice(1).map((t, i) => t - restTimes[i]).join(",");
  return sha256(new TextEncoder().encode(deltas));
}

// Slow-start handshake -------------------------------------------------
const requiredBreaths = 5;
let peerBreathCount = 0;

function handleBipPacket(pkt) {
  if (!pkt.phase || !pkt.agent_id) return;

  if (peerBreathCount < requiredBreaths) {
    if (pkt.phase === "REST") peerBreathCount++;
    if (peerBreathCount === requiredBreaths) {
      console.log(`Peer ${pkt.agent_id} accepted after ${requiredBreaths} breaths`);
    }
    return;          // drop all non-REST traffic until synced
  }

  // … normal symbol handling here …
}

// wire to WebSocket / UDP in appropriate environment
// frontendShellBreath.js  (pseudocode – not yet wired to any framework)
import { sha256 } from "./tinySha.js";   // any 200-B hash util
const REST_WINDOW = 256;
let restTimes = [];

function onRestPhase() {
  restTimes.push(Date.now());
  if (restTimes.length > REST_WINDOW) restTimes.shift();
}

function currentDigest() {
  if (restTimes.length < 2) return "0".repeat(64);
  const deltas = restTimes.slice(1).map((t, i) => t - restTimes[i]).join(",");
  return sha256(new TextEncoder().encode(deltas));
}

// Slow-start handshake -------------------------------------------------
const requiredBreaths = 5;
let peerBreathCount = 0;

function handleBipPacket(pkt) {
  if (!pkt.phase || !pkt.agent_id) return;

  if (peerBreathCount < requiredBreaths) {
    if (pkt.phase === "REST") peerBreathCount++;
    if (peerBreathCount === requiredBreaths) {
      console.log(`Peer ${pkt.agent_id} accepted after ${requiredBreaths} breaths`);
    }
    return;          // drop all non-REST traffic until synced
  }

  // … normal symbol handling here …
}

// wire to WebSocket / UDP in appropriate environment
