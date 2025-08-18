import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { 
  Search, 
  Edit,
  Eye,
  Calendar,
  User,
  Clock,
  Filter
} from "lucide-react";

const mockForms = [
  {
    id: "OASIS-001",
    patientName: "Margaret Johnson",
    visitDate: "2024-01-15",
    submitDate: "2024-01-15",
    status: "Completed",
    visitType: "Initial Assessment",
    completionTime: "45 min"
  },
  {
    id: "OASIS-002", 
    patientName: "Robert Chen",
    visitDate: "2024-01-14",
    submitDate: "2024-01-14",
    status: "Completed",
    visitType: "Follow-up",
    completionTime: "30 min"
  },
  {
    id: "OASIS-003",
    patientName: "Dorothy Williams",
    visitDate: "2024-01-12",
    submitDate: "2024-01-13",
    status: "Completed",
    visitType: "Discharge Assessment",
    completionTime: "38 min"
  },
  {
    id: "OASIS-004",
    patientName: "James Martinez",
    visitDate: "2024-01-10",
    submitDate: "2024-01-10",
    status: "Pending Review",
    visitType: "Initial Assessment",
    completionTime: "42 min"
  },
  {
    id: "OASIS-005",
    patientName: "Helen Davis",
    visitDate: "2024-01-08",
    submitDate: "2024-01-08",
    status: "Completed",
    visitType: "Follow-up",
    completionTime: "25 min"
  }
];

const FormsPage = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const navigate = useNavigate();

  const filteredForms = mockForms.filter((form) => {
    const matchesSearch = form.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         form.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === "all" || form.status.toLowerCase().includes(statusFilter.toLowerCase());
    
    return matchesSearch && matchesStatus;
  });

  const getStatusBadge = (status: string) => {
    const variants = {
      "Completed": "default",
      "Pending Review": "secondary",
      "Draft": "outline"
    } as const;
    
    return <Badge variant={variants[status as keyof typeof variants] || "outline"}>{status}</Badge>;
  };

  const handleEdit = (formId: string) => {
    navigate("/filling", { state: { formId } });
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
            <div>
              <h1 className="text-2xl font-bold text-foreground">Submitted Forms</h1>
              <p className="text-muted-foreground">Manage and review your OASIS assessments</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <Badge variant="outline" className="flex items-center space-x-1">
                <span>{filteredForms.length} forms</span>
              </Badge>
            </div>
          </div>

          {/* Filters */}
          <Card className="p-4">
            <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by patient name or form ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-muted-foreground" />
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Filter by status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="pending">Pending Review</SelectItem>
                    <SelectItem value="draft">Draft</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </Card>

          {/* Forms List - Mobile/Tablet */}
          <div className="block lg:hidden">
            <Card className="divide-y">
              {filteredForms.map((form) => (
                <div key={form.id} className="p-4 flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">{form.patientName}</span>
                      {getStatusBadge(form.status)}
                    </div>
                    <div className="text-sm text-muted-foreground space-y-1">
                      <div className="flex items-center gap-2">
                        <User className="h-4 w-4" />
                        <span className="truncate">ID: {form.id}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        <span>{new Date(form.visitDate).toLocaleDateString()}</span>
                        <span className="mx-1">•</span>
                        <span className="truncate">{form.visitType}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4" />
                        <span>{form.completionTime}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col sm:flex-row gap-2 shrink-0">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleEdit(form.id)}
                      className="inline-flex items-center gap-1"
                    >
                      <Edit className="h-4 w-4" />
                      <span>Edit</span>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="inline-flex items-center gap-1"
                    >
                      <Eye className="h-4 w-4" />
                      <span>View</span>
                    </Button>
                  </div>
                </div>
              ))}
            </Card>
          </div>

          {/* Forms Table - Desktop */}
          <div className="hidden lg:block">
            <Card className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Form ID</TableHead>
                    <TableHead>Patient Name</TableHead>
                    <TableHead>Visit Date</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredForms.map((form) => (
                    <TableRow key={form.id}>
                      <TableCell className="font-medium">{form.id}</TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <User className="h-4 w-4 text-muted-foreground" />
                          <span>{form.patientName}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Calendar className="h-4 w-4 text-muted-foreground" />
                          <span>{new Date(form.visitDate).toLocaleDateString()}</span>
                        </div>
                      </TableCell>
                      <TableCell>{form.visitType}</TableCell>
                      <TableCell>{getStatusBadge(form.status)}</TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          <span>{form.completionTime}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleEdit(form.id)}
                            className="flex items-center space-x-1"
                          >
                            <Edit className="h-4 w-4" />
                            <span>Edit</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex items-center space-x-1"
                          >
                            <Eye className="h-4 w-4" />
                            <span>View</span>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </div>

          {filteredForms.length === 0 && (
            <Card className="p-8 text-center">
              <div className="text-muted-foreground">
                <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No forms found</h3>
                <p>Try adjusting your search criteria or create a new assessment.</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default FormsPage;