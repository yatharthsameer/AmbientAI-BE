import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

const mockTranscript = [
  { speaker: "Nurse", time: "10:15", text: "Good morning, Mrs. Johnson. How are you feeling today?" },
  { speaker: "Patient", time: "10:15", text: "I'm doing okay, but my back has been hurting quite a bit. I'd say it's about a 7 out of 10." },
  { speaker: "Nurse", time: "10:16", text: "I'm sorry to hear that. Are you taking your prescribed pain medication?" },
  { speaker: "Patient", time: "10:16", text: "Yes, I take the ibuprofen twice a day as you told me, and the blood pressure medicine every morning." },
  { speaker: "Nurse", time: "10:17", text: "That's good. Let me check your blood pressure today. Can you sit up straight for me?" },
  { speaker: "Patient", time: "10:18", text: "Of course. I've been using my walker to get around the house safely." },
  { speaker: "Nurse", time: "10:19", text: "Your blood pressure is 140 over 85 today. How have you been managing with bathing and dressing?" },
  { speaker: "Patient", time: "10:20", text: "I can dress myself, but I need help with showering. My daughter comes by twice a week to help." },
];

// Map answers to transcript sources for highlighting
const answerToTranscriptMap: Record<string, string[]> = {
  age: ["78"],
  diagnosis: ["back has been hurting", "blood pressure"],
  medications: ["ibuprofen twice a day", "blood pressure medicine every morning"],
  allergies: ["None reported"],
  emergency_contact: ["daughter"],
  mobility: ["walker to get around the house safely"],
  adl_bathing: ["need help with showering", "daughter comes by twice a week"],
  adl_dressing: ["I can dress myself"],
  fall_risk: ["walker"],
  cognitive: ["Alert"],
  pain_level: ["7 out of 10"],
  vital_signs: ["140 over 85"],
  wound_care: ["No wounds"],
  nutrition: ["Adequate"],
  elimination: ["regular"],
  home_safety: ["walker", "daughter"],
  support_system: ["daughter comes by twice a week"],
  equipment: ["walker"],
  medication_management: ["I take the ibuprofen"],
  emergency_plan: ["daughter"],
  goals: ["back has been hurting"],
  interventions: ["pain medication"],
  education: ["medication"],
  next_visit: ["Follow-up"],
  physician_communication: ["blood pressure"]
};

const oasisSections = [
  {
    id: "demographics",
    title: "Patient Demographics & History",
    questions: [
      { id: "age", label: "Patient Age", value: "78", type: "input" },
      { id: "diagnosis", label: "Primary Diagnosis", value: "Hypertension, Chronic back pain", type: "textarea" },
      { id: "medications", label: "Current Medications", value: "Ibuprofen 400mg BID, Lisinopril 10mg daily", type: "textarea" },
      { id: "allergies", label: "Known Allergies", value: "None reported", type: "input" },
      { id: "emergency_contact", label: "Emergency Contact", value: "Daughter - Sarah Johnson", type: "input" },
    ]
  },
  {
    id: "functional_status",
    title: "Functional Status",
    questions: [
      { id: "mobility", label: "Mobility Assessment", value: "Uses walker for ambulation", type: "textarea" },
      { id: "adl_bathing", label: "Bathing", value: "Requires assistance - daughter helps 2x/week", type: "input" },
      { id: "adl_dressing", label: "Dressing", value: "Independent", type: "input" },
      { id: "fall_risk", label: "Fall Risk Assessment", value: "Moderate - uses walker appropriately", type: "textarea" },
      { id: "cognitive", label: "Cognitive Status", value: "Alert and oriented x3", type: "input" },
    ]
  },
  {
    id: "clinical_status",
    title: "Clinical Status",
    questions: [
      { id: "pain_level", label: "Pain Assessment", value: "7/10 chronic back pain", type: "input" },
      { id: "vital_signs", label: "Vital Signs", value: "BP: 140/85, HR: 78, Temp: 98.6°F", type: "input" },
      { id: "wound_care", label: "Wound/Skin Assessment", value: "No wounds noted, skin intact", type: "textarea" },
      { id: "nutrition", label: "Nutritional Status", value: "Adequate oral intake", type: "input" },
      { id: "elimination", label: "Bowel/Bladder Function", value: "Continent, regular bowel movements", type: "input" },
    ]
  },
  {
    id: "safety_environment",
    title: "Safety & Environment",
    questions: [
      { id: "home_safety", label: "Home Safety Assessment", value: "Walker available, daughter provides support", type: "textarea" },
      { id: "support_system", label: "Support System", value: "Daughter visits 2x/week, neighbors check in", type: "textarea" },
      { id: "equipment", label: "DME/Equipment Needs", value: "Walker in good condition", type: "input" },
      { id: "medication_management", label: "Medication Management", value: "Patient manages independently", type: "input" },
      { id: "emergency_plan", label: "Emergency Plan", value: "Emergency contact information posted", type: "textarea" },
    ]
  },
  {
    id: "plan_goals",
    title: "Care Plan & Goals",
    questions: [
      { id: "goals", label: "Patient Goals", value: "Manage pain, maintain independence", type: "textarea" },
      { id: "interventions", label: "Planned Interventions", value: "Pain management education, safety assessment", type: "textarea" },
      { id: "education", label: "Patient Education", value: "Medication compliance, fall prevention", type: "textarea" },
      { id: "next_visit", label: "Next Visit Plan", value: "Follow-up in 1 week", type: "input" },
      { id: "physician_communication", label: "Physician Communication", value: "Will contact regarding BP elevation", type: "textarea" },
    ]
  }
];

const FillingPage = () => {
  const [highlightedPhrases, setHighlightedPhrases] = useState<string[]>([]);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const { toast } = useToast();

  const handleAnswerClick = (questionId: string, value: string) => {
    const phrases = answerToTranscriptMap[questionId] || [];
    setHighlightedPhrases(phrases);
    // Clear highlights after 5 seconds
    setTimeout(() => setHighlightedPhrases([]), 5000);
  };

  const handleSubmit = () => {
    toast({
      title: "Form Submitted Successfully",
      description: "OASIS form has been saved and submitted for review.",
    });
  };

  const updateFormData = (questionId: string, value: string) => {
    setFormData(prev => ({ ...prev, [questionId]: value }));
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-foreground">OASIS Form Completion</h1>
          <p className="text-muted-foreground">Review and complete the assessment form</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left Column - OASIS Form */}
          <div className="space-y-4">
            <Card className="p-6">
              <Accordion type="multiple" className="w-full">
                {oasisSections.map((section) => (
                  <AccordionItem key={section.id} value={section.id}>
                    <AccordionTrigger className="text-left font-semibold">
                      {section.title}
                    </AccordionTrigger>
                    <AccordionContent className="space-y-4">
                      {section.questions.map((question) => (
                        <div key={question.id} className="space-y-2">
                          <Label htmlFor={question.id} className="text-sm font-medium">
                            {question.label}
                          </Label>
                          <div className="relative">
                            {question.type === "textarea" ? (
                              <Textarea
                                id={question.id}
                                value={formData[question.id] || question.value}
                                onChange={(e) => updateFormData(question.id, e.target.value)}
                                onClick={() => handleAnswerClick(question.id, question.value)}
                                className="cursor-pointer hover:bg-medical-blue-light/50 transition-colors"
                                rows={3}
                              />
                            ) : (
                              <Input
                                id={question.id}
                                value={formData[question.id] || question.value}
                                onChange={(e) => updateFormData(question.id, e.target.value)}
                                onClick={() => handleAnswerClick(question.id, question.value)}
                                className="cursor-pointer hover:bg-medical-blue-light/50 transition-colors"
                              />
                            )}
                            <Badge 
                              variant="secondary" 
                              className="absolute -top-2 -right-2 text-xs"
                            >
                              Auto-filled
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </Card>

            <Button 
              onClick={handleSubmit}
              size="lg"
              className="w-full"
            >
              Submit OASIS Form
            </Button>
          </div>

          {/* Right Column - Transcript */}
          <div className="space-y-4 md:sticky md:top-16 self-start">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Conversation Transcript</h3>
              <div className="space-y-4 max-h-[60vh] md:max-h-[70vh] overflow-y-auto">
                {mockTranscript.map((entry, index) => (
                  <div key={index} className="flex space-x-3">
                    <div className="flex-shrink-0">
                      <Badge 
                        variant={entry.speaker === "Nurse" ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {entry.speaker}
                      </Badge>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-xs text-muted-foreground">{entry.time}</span>
                      </div>
                      <p className="text-sm leading-relaxed">
                        {highlightedPhrases.length > 0 ? (
                          entry.text.split(' ').map((word, wordIndex) => {
                            const isHighlighted = highlightedPhrases.some(phrase => 
                              phrase.toLowerCase().includes(word.toLowerCase()) ||
                              word.toLowerCase().includes(phrase.toLowerCase())
                            );
                            return (
                              <span
                                key={wordIndex}
                                className={isHighlighted ? 'bg-destructive/20 border-b-2 border-destructive' : ''}
                              >
                                {word}{wordIndex < entry.text.split(' ').length - 1 ? ' ' : ''}
                              </span>
                            );
                          })
                        ) : (
                          entry.text
                        )}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FillingPage;