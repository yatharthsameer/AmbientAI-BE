import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts";
import {
  Clock,
  FileText,
  TrendingUp,
  Users,
  CheckCircle,
  AlertTriangle
} from "lucide-react";

const weeklyData = [
  { week: "Week 1", visits: 12, completionTime: 35 },
  { week: "Week 2", visits: 15, completionTime: 32 },
  { week: "Week 3", visits: 18, completionTime: 28 },
  { week: "Week 4", visits: 14, completionTime: 30 }
];

const monthlyForms = [
  { month: "Nov", forms: 45 },
  { month: "Dec", forms: 52 },
  { month: "Jan", forms: 48 }
];

const missedQuestions = [
  { question: "Fall Risk Assessment", count: 8, color: "#e74c3c" },
  { question: "Pain Scale Documentation", count: 6, color: "#f39c12" },
  { question: "Medication Reconciliation", count: 4, color: "#3498db" },
  { question: "ADL Assessment", count: 3, color: "#2ecc71" }
];

const COLORS = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71'];

const DashboardPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6">
        <div className="space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-2xl font-bold text-foreground">Performance Dashboard</h1>
            <p className="text-muted-foreground">Track your assessment efficiency and outcomes</p>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Clock className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg. Visit Time</p>
                  <p className="text-2xl font-bold">31 min</p>
                  <Badge variant="secondary" className="text-xs mt-1">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    15% improvement
                  </Badge>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-success/10 rounded-lg">
                  <FileText className="h-6 w-6 text-success" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Forms This Month</p>
                  <p className="text-2xl font-bold">48</p>
                  <Badge variant="secondary" className="text-xs mt-1">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    +12% vs last month
                  </Badge>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <Users className="h-6 w-6 text-accent" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Patients Seen</p>
                  <p className="text-2xl font-bold">64</p>
                  <Badge variant="secondary" className="text-xs mt-1">
                    This month
                  </Badge>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-medical-green/10 rounded-lg">
                  <CheckCircle className="h-6 w-6 text-medical-green" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Completion Rate</p>
                  <p className="text-2xl font-bold">94%</p>
                  <Badge variant="secondary" className="text-xs mt-1">
                    Quality score
                  </Badge>
                </div>
              </div>
            </Card>
          </div>

          {/* Charts Row 1 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Weekly Performance */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Weekly Performance Trends</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="week" 
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px"
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="visits" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      name="Visits"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="completionTime" 
                      stroke="hsl(var(--accent))" 
                      strokeWidth={2}
                      name="Avg. Time (min)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Monthly Forms */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Monthly Form Submissions</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={monthlyForms}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="month" 
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px"
                      }}
                    />
                    <Bar 
                      dataKey="forms" 
                      fill="hsl(var(--primary))" 
                      radius={[4, 4, 0, 0]}
                      name="Forms Completed"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Charts Row 2 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Missed Questions Analysis */}
            <Card className="p-6 lg:col-span-2">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Frequently Missed Questions</h3>
                <Badge variant="outline" className="flex items-center space-x-1">
                  <AlertTriangle className="h-3 w-3" />
                  <span>Quality Focus Areas</span>
                </Badge>
              </div>
              <div className="space-y-4">
                {missedQuestions.map((item, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm font-medium">{item.question}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="secondary">{item.count} times</Badge>
                      <div className="w-24 bg-muted rounded-full h-2">
                        <div 
                          className="h-2 rounded-full"
                          style={{ 
                            backgroundColor: item.color,
                            width: `${(item.count / 8) * 100}%`
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Quality Metrics */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Quality Distribution</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={missedQuestions}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="count"
                    >
                      {missedQuestions.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px"
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;