import { useLocation, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { 
  Mic,
  ClipboardList,
  FileText,
  BarChart3,
  Stethoscope
} from "lucide-react";

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    {
      path: "/",
      label: "Record",
      icon: Mic,
    },
    {
      path: "/filling",
      label: "Fill Form",
      icon: ClipboardList,
    },
    {
      path: "/forms",
      label: "Forms",
      icon: FileText,
    },
    {
      path: "/dashboard",
      label: "Dashboard",
      icon: BarChart3,
    },
  ];

  return (
    <nav className="bg-card border-b border-border shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <Stethoscope className="h-8 w-8 text-primary" />
            <span className="text-xl font-semibold text-foreground">
              HealthScribe
            </span>
          </div>
          
          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Button
                  key={item.path}
                  asChild
                  variant={isActive ? "default" : "ghost"}
                  size="sm"
                  className="flex items-center space-x-2"
                >
                  <Link to={item.path}>
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </Link>
                </Button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;