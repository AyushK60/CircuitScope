export default function MetricCard({ name, value, sub }: { name: string; value: string; sub: string }) {
  return (
    <div className="metric-card">
      <div className="name">{name}</div>
      <div className="value">{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}
