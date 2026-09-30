import { useEffect, useMemo, useState } from 'react'

const API = 'http://127.0.0.1:8000'
const example = 'My card was charged twice and I do not recognize the transaction. Connect me to a human agent.'
const darkThemeCss = `
body.dark-body{background:#080d18!important}.dark-mode{color:#e8eefc}.dark-mode nav a,.dark-mode .brand{color:#e8eefc}.dark-mode .composer,.dark-mode .queue-panel,.dark-mode .result,.dark-mode .blank,.dark-mode .metric{background:#111a2d;border-color:#273653;box-shadow:none}.dark-mode .panel-heading h2,.dark-mode .queue-panel h2,.dark-mode .metric strong,.dark-mode .blank h2,.dark-mode .reasoning h3,.dark-mode .evidence h3{color:#f3f6ff}.dark-mode .composer label,.dark-mode .reasoning li,.dark-mode .ticket p,.dark-mode .source b{color:#c2cee5}.dark-mode .composer textarea,.dark-mode input{background:#0c1424;color:#edf2ff;border-color:#344564}.dark-mode .result-banner{background:linear-gradient(90deg,#121d34,#172847)}.dark-mode .evidence{border-color:#2a3852}.dark-mode .evidence blockquote{background:#0c1527;border-color:#2f456c;color:#d6e2f8}.dark-mode .audit{background:#0d1423}.dark-mode .ticket{border-color:#2b3954}.dark-mode .ticket footer,.dark-mode .reasoning li,.dark-mode .source{border-color:#27344c}.dark-mode .bar{background:#24314a}.dark-mode .subtle{color:#aebbd2}.dark-mode footer{border-color:#2a3650}.theme-toggle{border:1px solid #b8c8e9;border-radius:99px;padding:6px 9px;background:#fff;color:#155eef;font-weight:800;cursor:pointer}.dark-mode .theme-toggle{background:#17243c;color:#ffe49a;border-color:#40577e}`

const labels = {
  critical: ['Critical', 'critical'], high: ['High', 'high'], medium: ['Medium', 'medium'], low: ['Low', 'low'],
}

function Badge({ priority }) {
  const [label, tone] = labels[priority] || [priority, 'medium']
  return <span className={`badge ${tone}`}><i />{label} priority</span>
}

function Metric({ label, value, muted }) {
  return <div className="metric"><span>{label}</span><strong className={muted ? 'muted' : ''}>{value}</strong></div>
}

function Entities({ entities }) {
  const labels = { order_ids: 'Order IDs', transaction_ids: 'Transaction IDs', card_references: 'Card references', amounts: 'Amounts', dates: 'Dates' }
  const found = Object.entries(entities || {}).filter(([, values]) => values?.length)
  return <div className="sources"><p className="eyebrow">Detected entities</p>{found.length ? found.map(([type, values]) => <div className="source" key={type}><span>{labels[type] || type}</span><b>{values.join(', ')}</b><small>extracted</small></div>) : <p className="subtle">No operational references detected in this message.</p>}</div>
}

function Evaluation({ metrics }) {
  const score = value => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
  return <section className="queue-panel" id="evaluation"><div className="panel-heading"><div><p className="eyebrow">Evaluation evidence</p><h2>Model quality on held-out data</h2></div><span className="live-dot">Reproducible</span></div>{!metrics ? <p className="subtle">Evaluation files are not available yet. Run the evaluation scripts from the project root.</p> : <div className="stat-strip"><Metric label="Baseline macro F1" value={score(metrics.baseline?.macro_f1)} /><Metric label="DistilBERT macro F1" value={score(metrics.transformer?.macro_f1)} /><Metric label="Retrieval Recall@5" value={score(metrics.retrieval?.recall_at_5)} /><Metric label="Safety escalation F1" value={score(metrics.safety?.escalation_f1)} /></div>}</section>
}

function CustomerReceipt({ result }) {
  const message = result.escalate_to_human
    ? 'Your request has been sent to a specialist for secure review.'
    : 'Your request has been received and routed to the appropriate support team.'
  return <section className="blank" aria-live="polite"><span>✓</span><p className="eyebrow">Ticket received</p><h2>{message}</h2><p>Reference: <code>{result.ticket?.id || 'processing'}</code></p><p>For security, detailed internal routing and policy evidence are visible only in the agent console.</p></section>
}

function AgentLoginForm({ onLogin, onBack }) {
  const [accessKey, setAccessKey] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const submit = async event => { event.preventDefault(); setLoading(true); setError(''); try { await onLogin(accessKey) } catch (err) { setError(err.message) } finally { setLoading(false) } }
  return <section className="blank"><span>⌁</span><p className="eyebrow">Agent access</p><h2>Sign in to the Agent Console</h2><p>Customer requests and internal triage evidence are separated.</p><form onSubmit={submit} style={{ maxWidth: 420, margin: '22px auto 0' }}><input type="password" value={accessKey} onChange={event => setAccessKey(event.target.value)} placeholder="Agent access key" minLength="8" required style={{ width: '100%', padding: 12, border: '1px solid #d0d5dd', borderRadius: 10 }} /><button style={{ width: '100%', marginTop: 10, padding: 12, border: 0, borderRadius: 10, background: '#155eef', color: '#fff', fontWeight: 700, cursor: 'pointer' }} disabled={loading}>{loading ? 'Verifying…' : 'Sign in as agent'}</button></form>{error && <p className="error" style={{ maxWidth: 420, margin: '12px auto 0' }}>{error}</p>}<button className="text-button" type="button" style={{ marginTop: 16 }} onClick={onBack}>Return to Customer Portal</button></section>
}

function Queue({ tickets, loading, onResolve, resolvingId, outcomes, onOutcomeChange, onInspect }) {
  return <section className="queue-panel">
    <div className="panel-heading"><div><p className="eyebrow">Agent workspace</p><h2>Active ticket queue</h2></div><span className="live-dot">Live</span></div>
    {loading ? <p className="subtle">Loading tickets…</p> : tickets.length === 0 ? <div className="empty"><span>⌁</span><p>No active tickets in the queue.</p></div> :
      <div className="ticket-list">{tickets.slice(0, 5).map(ticket => <article className="ticket" key={ticket.id}><div className="ticket-top"><Badge priority={ticket.priority} /><small>{ticket.intent.replaceAll('_', ' ')}</small></div><p>{ticket.customer_message}</p><footer><span>{ticket.department}</span><span>{ticket.status.replace('_', ' ')}</span></footer><button className="text-button" type="button" style={{ marginTop: 10 }} onClick={() => onInspect(ticket)}>View NLP analysis</button><textarea aria-label={`Outcome note for ticket ${ticket.id}`} value={outcomes[ticket.id] || ''} onChange={event => onOutcomeChange(ticket.id, event.target.value)} placeholder="Agent outcome note (optional)" maxLength="3000" style={{ width: '100%', minHeight: 46, marginTop: 10, padding: 7, border: '1px solid #dbe3ef', borderRadius: 7, color: '#344054', fontSize: '.65rem', resize: 'vertical' }} /><button className="resolve-button" style={{ width: '100%', marginTop: 8, padding: '7px', border: '1px solid #bed1ff', borderRadius: 7, background: '#f4f8ff', color: '#155eef', fontSize: '.64rem', fontWeight: 700, cursor: 'pointer' }} onClick={() => onResolve(ticket, outcomes[ticket.id])} disabled={resolvingId === ticket.id}>{resolvingId === ticket.id ? 'Saving…' : 'Mark resolved ✓'}</button></article>)}</div>}
  </section>
}

function TicketInspector({ ticket, onClose }) {
  const sources = ticket.retrieved_sources || []
  return <section className="result" aria-live="polite"><div className="result-banner"><div><p className="eyebrow">Selected agent ticket</p><h2>Stored NLP analysis</h2></div><div><Badge priority={ticket.priority} /> <button className="text-button" type="button" style={{ marginLeft: 10 }} onClick={onClose}>Close</button></div></div><div className="metrics-grid"><Metric label="Query type" value={ticket.query_type?.replaceAll('_', ' ') || 'other'} /><Metric label="Fine intent" value={ticket.intent.replaceAll('_', ' ')} /><Metric label="Confidence" value={`${Math.round(ticket.confidence * 100)}%`} muted={ticket.confidence < .55} /><Metric label="Sentiment" value={ticket.sentiment} /></div>{ticket.escalated && <div className="alert"><b>Human review required</b><span>{ticket.escalation_reasons.join(' · ')}</span></div>}<div className="result-columns"><div className="reasoning"><p className="eyebrow">Customer message</p><h3>Stored, masked ticket</h3><blockquote>{ticket.customer_message}</blockquote><Entities entities={ticket.entities} /></div><div className="evidence"><p className="eyebrow">Retrieved evidence</p><h3>Policy sources</h3><div className="sources">{sources.length ? sources.map(source => <div className="source" key={source.id}><span>{source.id}</span><b>{source.title}</b><small>{Math.round(source.score * 100)}% match</small></div>) : <p className="subtle">No policy sources were stored.</p>}</div>{ticket.agent_outcome && <><p className="eyebrow" style={{ marginTop: 22 }}>Agent outcome</p><blockquote>{ticket.agent_outcome}</blockquote></>}</div></div></section>
}

function Result({ result }) {
  const top = result.top_intents?.[0]
  const displayedSources = result.retrieved_sources || []
  return <section className="result" aria-live="polite">
    <div className="result-banner"><div><p className="eyebrow">Triage decision</p><h2>Ticket is ready for its next action.</h2></div><Badge priority={result.priority} /></div>
    <div className="metrics-grid">
      <Metric label="Query type" value={result.query_type?.replaceAll('_', ' ') || 'other'} />
      <Metric label="Query confidence" value={result.query_type_confidence == null ? '—' : `${Math.round(result.query_type_confidence * 100)}%`} muted={result.query_type_confidence != null && result.query_type_confidence < .55} />
      <Metric label="Fine intent" value={result.intent.replaceAll('_', ' ')} />
      <Metric label="Route" value={result.department} />
      <Metric label="Confidence" value={`${Math.round(result.confidence * 100)}%`} muted={result.confidence < .55} />
      <Metric label="Sentiment" value={`${result.sentiment} · ${result.sentiment_score.toFixed(2)}`} />
    </div>
    {result.escalate_to_human ? <div className="alert"><b>Human review required</b><span>{result.escalation_reasons.join(' · ')}</span></div> : <div className="success">✓ Safe for automatic specialist routing</div>}
    {result.agent_handoff && <div className="success"><b>Structured agent handoff</b><span> {result.agent_handoff.summary}</span></div>}
    <div className="result-columns">
      <div className="reasoning"><p className="eyebrow">Decision signals</p><h3>Why this route</h3><ul>{result.explanation.map((item, index) => <li key={index}>{item}</li>)}</ul>
        <div className="intent-bars"><p className="eyebrow">Intent confidence</p>{result.top_intents.map(item => <div className="bar-row" key={item.intent}><span>{item.intent.replaceAll('_', ' ')}</span><div className="bar"><i style={{ width: `${item.confidence * 100}%` }} /></div><b>{Math.round(item.confidence * 100)}%</b></div>)}</div><Entities entities={result.entities} />
      </div>
      <div className="evidence"><p className="eyebrow">Grounded next step</p><h3>Policy-backed response</h3><blockquote>{result.suggested_response.answer}</blockquote><div className="sources"><p className="eyebrow">Retrieved evidence · {result.retrieval_backend}</p>{displayedSources.map(source => <div className="source" key={source.id}><span>{source.id}</span><b>{source.title}</b><small>{Math.round(source.score * 100)}% match</small></div>)}</div></div>
    </div>
    {result.ticket && <p className="audit">Ticket saved · <code>{result.ticket.id}</code></p>}
  </section>
}

export default function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState(null)
  const [tickets, setTickets] = useState([])
  const [loading, setLoading] = useState(false)
  const [queueLoading, setQueueLoading] = useState(true)
  const [resolvingId, setResolvingId] = useState(null)
  const [outcomes, setOutcomes] = useState({})
  const [metrics, setMetrics] = useState(null)
  const [view, setView] = useState('customer')
  const [agentKey, setAgentKey] = useState(() => sessionStorage.getItem('resolveai_agent_key') || '')
  const [darkTheme, setDarkTheme] = useState(() => sessionStorage.getItem('resolveai_theme') === 'dark')
  const [selectedTicket, setSelectedTicket] = useState(null)
  const [error, setError] = useState('')

  const activeTickets = useMemo(() => tickets.filter(ticket => ticket.status !== 'resolved'), [tickets])
  const stats = useMemo(() => ({ total: activeTickets.length, critical: activeTickets.filter(t => t.priority === 'critical').length, escalated: activeTickets.filter(t => t.escalated).length }), [activeTickets])
  const agentHeaders = () => ({ 'X-Agent-Key': agentKey })
  const loadTickets = async () => { if (!agentKey) { setQueueLoading(false); return } setQueueLoading(true); try { const response = await fetch(`${API}/tickets?limit=20`, { headers: agentHeaders() }); if (!response.ok) throw new Error(); setTickets(await response.json()) } catch { setError('Agent session could not load ticket data. Sign in again or check the API.') } finally { setQueueLoading(false) } }
  const loadMetrics = async () => { if (!agentKey) return; try { const response = await fetch(`${API}/metrics`, { headers: agentHeaders() }); if (response.ok) setMetrics(await response.json()) } catch { /* Metrics are optional to normal ticket analysis. */ } }
  useEffect(() => { if (agentKey) { loadTickets(); loadMetrics() } else { setQueueLoading(false); setTickets([]); setMetrics(null) } }, [agentKey])
  useEffect(() => { document.body.classList.toggle('dark-body', darkTheme); sessionStorage.setItem('resolveai_theme', darkTheme ? 'dark' : 'light') }, [darkTheme])
  const analyse = async event => { event.preventDefault(); if (message.trim().length < 3) return setError('Please enter a complete customer message.'); setLoading(true); setError(''); try { const response = await fetch(`${API}/tickets/analyse`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, save: true }) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || 'Analysis failed.'); setResult(body); await loadTickets() } catch (err) { setError(err.message.includes('fetch') ? 'The FastAPI service is unavailable. Start it on port 8000.' : err.message) } finally { setLoading(false) } }
  const resolveTicket = async (ticket, outcome) => { setResolvingId(ticket.id); setError(''); try { const response = await fetch(`${API}/tickets/${ticket.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json', ...agentHeaders() }, body: JSON.stringify({ status: 'resolved', agent_outcome: outcome?.trim() || 'Resolved from the agent workspace.' }) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || 'Could not resolve ticket.'); setOutcomes(current => { const next = { ...current }; delete next[ticket.id]; return next }); await loadTickets() } catch (err) { setError(err.message.includes('fetch') ? 'The FastAPI service is unavailable. Start it on port 8000.' : err.message) } finally { setResolvingId(null) } }
  const loginAgent = async accessKey => { const response = await fetch(`${API}/agent/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ access_key: accessKey }) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || 'Agent sign-in failed.'); sessionStorage.setItem('resolveai_agent_key', accessKey); setAgentKey(accessKey); setView('agent') }
  const openAgentView = target => setView(agentKey ? target : 'login')
  const logoutAgent = () => { sessionStorage.removeItem('resolveai_agent_key'); setAgentKey(''); setView('customer') }

  return <main className={darkTheme ? 'dark-mode' : ''}>
    <style>{darkThemeCss}</style><nav><a className="brand" href="#top" onClick={() => setView('customer')}><span>R</span> ResolveAI</a><div><a href="#customer" onClick={() => setView('customer')}>Customer portal</a><a href="#agent" onClick={() => openAgentView('agent')}>Agent console</a><a href="#evaluation" onClick={() => openAgentView('quality')}>Model quality</a>{agentKey && <a href="#customer" onClick={logoutAgent}>Sign out</a>}<button className="theme-toggle" type="button" onClick={() => setDarkTheme(current => !current)} aria-label="Toggle color theme">{darkTheme ? '☀ Light' : '☾ Dark'}</button><span className="system">SYSTEM ONLINE</span></div></nav>
    <header className="hero" id="top"><div><p className="eyebrow">{view === 'customer' ? 'Customer support portal' : view === 'agent' ? 'Restricted agent workspace' : 'Evaluation workspace'}</p><h1>{view === 'customer' ? <>Get help with<br /><em>confidence.</em></> : view === 'agent' ? <>Resolve every<br /><em>important signal.</em></> : <>Measure what<br /><em>matters.</em></>}</h1><p className="lead">{view === 'customer' ? 'Submit a banking-support request. Sensitive and uncertain cases are securely assigned to a specialist.' : view === 'agent' ? 'Review routed tickets, inspect decision evidence, and record human outcomes.' : 'Inspect reproducible held-out evaluation results for every NLP component.'}</p><div className="hero-stats"><span><b>77</b> intent classes</span><span><b>4</b> priority levels</span><span><b>∞</b> human judgement</span></div></div><div className="hero-orb"><span>◎</span><small>TRIAGE<br />ENGINE</small></div></header>
    {view === 'customer' && <><section className="workspace" id="customer"><div className="composer"><div className="panel-heading"><div><p className="eyebrow">New support ticket</p><h2>How can we help?</h2></div><button className="text-button" type="button" onClick={() => setMessage(example)}>Use example</button></div><form onSubmit={analyse}><label htmlFor="message">Your message</label><textarea id="message" value={message} onChange={event => setMessage(event.target.value)} placeholder="Describe your banking-support issue…" maxLength="5000" /><div className="form-footer"><span>{message.length}/5000 characters</span><button disabled={loading}>{loading ? 'Sending request…' : 'Submit request'} <b>→</b></button></div></form>{error && <p className="error">{error}</p>}</div><aside className="guardrail"><p className="eyebrow">Customer protection</p><h3>Helpful. Never speculative.</h3><p>We do not ask for passwords, PINs, or full card numbers. Sensitive requests are reviewed by a human specialist.</p><ul><li>Suspected fraud → priority review</li><li>Uncertain request → human verification</li><li>Human request → agent assignment</li></ul></aside></section>{result ? <CustomerReceipt result={result} /> : <section className="blank"><span>R</span><p className="eyebrow">Secure support</p><h2>Tell us what you need help with.</h2><p>Your support request will be triaged and routed securely.</p></section>}</>}
    {view === 'login' && <AgentLoginForm onLogin={loginAgent} onBack={() => setView('customer')} />}
    {view === 'agent' && <><section className="stat-strip"><Metric label="Active tickets" value={stats.total} /><Metric label="Critical risk" value={stats.critical} /><Metric label="Human escalations" value={stats.escalated} /><Metric label="Decision mode" value="Grounded" /></section>{result && <Result result={result} />}{selectedTicket && <TicketInspector ticket={selectedTicket} onClose={() => setSelectedTicket(null)} />}<div id="agent"><Queue tickets={activeTickets} loading={queueLoading} onResolve={resolveTicket} resolvingId={resolvingId} outcomes={outcomes} onOutcomeChange={(ticketId, note) => setOutcomes(current => ({ ...current, [ticketId]: note }))} onInspect={setSelectedTicket} /></div></>}
    {view === 'quality' && <Evaluation metrics={metrics} />}
    <footer><span>ResolveAI · Explainable NLP customer-support triage</span><span>FastAPI · TF-IDF baseline · grounded retrieval</span></footer>
  </main>
}
