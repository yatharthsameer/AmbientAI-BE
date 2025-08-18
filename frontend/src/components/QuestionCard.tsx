import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X, Check } from "lucide-react";

interface QuestionCardProps {
  question: string;
  onDismiss: () => void;
}

const QuestionCard = ({ question, onDismiss }: QuestionCardProps) => {
  const [isVisible, setIsVisible] = useState(true);

  const handleDismiss = () => {
    setIsVisible(false);
    setTimeout(onDismiss, 200);
  };

  if (!isVisible) return null;

  return (
    <Card className="fixed top-20 left-1/2 transform -translate-x-1/2 w-11/12 max-w-md bg-card shadow-card border border-border z-50 transition-all duration-200 animate-in slide-in-from-top-2">
      <div className="p-4">
        <div className="flex items-start justify-between space-x-3">
          <div className="flex-1">
            <p className="text-sm font-medium text-warning mb-2">
              Missing Assessment Question
            </p>
            <p className="text-foreground text-sm leading-relaxed">
              {question}
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDismiss}
            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="mt-4 flex justify-end">
          <Button
            size="sm"
            onClick={handleDismiss}
            className="flex items-center space-x-1"
          >
            <Check className="h-3 w-3" />
            <span>Addressed</span>
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default QuestionCard;