// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
/// \file KstarToK0Gamma.cxx
/// \brief K*(892) --> K0 Gamma analysis task
///
/// \author David Dobrigkeit Chinellato <david.dobrigkeit.chinellato@cern.ch>, Austrian Academy of Sciences & SMI
/// \author Romain Schotter <romain.schotter@cern.ch>, Austrian Academy of Sciences & SMI
//
// V0 analysis task
// ================
//
// This code loops over a V0Cores table and produces some
// standard analysis output. It is meant to be run over
// derived data.
//
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    romain.schotter@cern.ch
//    david.dobrigkeit.chinellato@cern.ch
//

#include <Math/Vector4D.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <TPDGCode.h>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/O2DatabasePDGPlugin.h"
#include "ReconstructionDataFormats/Track.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/PhysicsConstants.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Multiplicity.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "PWGUD/Core/SGSelector.h"
#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

#include "EventFiltering/Zorro.h"
#include "EventFiltering/ZorroSummary.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using DauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;
using DauMCTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackMCIds, aod::DauTrackTPCPIDs>;
using V0Candidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0LambdaMLScores, aod::V0AntiLambdaMLScores, aod::V0K0ShortMLScores>;
// using V0MCCandidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0MCCores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCMothers, aod::V0MCCollRefs>;
using V0MCCandidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0Extras, aod::V0TOFPIDs, aod::V0TOFNSigmas, aod::V0MCMothers, aod::V0CoreMCLabels, aod::V0LambdaMLScores, aod::V0AntiLambdaMLScores, aod::V0K0ShortMLScores>;

struct KstarToK0Gamma {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // master analysis switches
  Configurable<bool> doPPAnalysis{"doPPAnalysis", true, "If running on pp collision, switch it on true"};
  Configurable<bool> doMCAssociation{"doMCAssociation", true, "if MC, do MC association"};

  // for running over skimmed dataset
  Configurable<bool> cfgSkimmedProcessing{"cfgSkimmedProcessing", false, "If running over skimmed data, switch it on true"};
  Configurable<std::string> cfgSkimmedTrigger{"cfgSkimmedTrigger", "fDoubleXi,fTripleXi,fQuadrupleXi", "(std::string) Comma separated list of triggers of interest"};

  // rapidity cut on the K(0, +, -)-Gamma pair
  Configurable<float> rapidityCut{"rapidityCut", 0.5, "rapidity cut on the K*(892)"};

  Configurable<bool> qaCentrality{"qaCentrality", false, "qa centrality flag: check base raw values"};  

  // UPC selections
  SGSelector sgSelector;
  struct : ConfigurableGroup {
    Configurable<float> fv0Cut{"upcCuts.fv0Cut", 100., "FV0A threshold"};
    Configurable<float> ft0aCut{"upcCuts.ft0aCut", 200., "FT0A threshold"};
    Configurable<float> ft0cCut{"upcCuts.ft0cCut", 100., "FT0C threshold"};
    Configurable<float> zdcCut{"upcCuts.zdcCut", 10., "ZDC threshold"};
    // Configurable<float> gapSel{"upcCuts.gapSel", 2, "Gap selection"};
  } upcCuts;

  // switch on/off event selections
  struct : ConfigurableGroup {
    Configurable<bool> requireSel8{"requireSel8", true, "require sel8 event selection"};
    Configurable<bool> requireTriggerTVX{"requireTriggerTVX", true, "require FT0 vertex (acceptable FT0C-FT0A time difference) at trigger level"};
    Configurable<bool> rejectITSROFBorder{"rejectITSROFBorder", true, "reject events at ITS ROF border"};
    Configurable<bool> rejectTFBorder{"rejectTFBorder", true, "reject events at TF border"};
    Configurable<bool> requireIsVertexITSTPC{"requireIsVertexITSTPC", false, "require events with at least one ITS-TPC track"};
    Configurable<bool> requireIsGoodZvtxFT0VsPV{"requireIsGoodZvtxFT0VsPV", true, "require events with PV position along z consistent (within 1 cm) between PV reconstructed using tracks and PV using FT0 A-C time difference"};
    Configurable<bool> requireIsVertexTOFmatched{"requireIsVertexTOFmatched", false, "require events with at least one of vertex contributors matched to TOF"};
    Configurable<bool> requireIsVertexTRDmatched{"requireIsVertexTRDmatched", false, "require events with at least one of vertex contributors matched to TRD"};
    Configurable<bool> rejectSameBunchPileup{"rejectSameBunchPileup", true, "reject collisions in case of pileup with another collision in the same foundBC"};
    Configurable<bool> requireNoCollInTimeRangeStd{"requireNoCollInTimeRangeStd", false, "reject collisions corrupted by the cannibalism, with other collisions within +/- 2 microseconds or mult above a certain threshold in -4 - -2 microseconds"};
    Configurable<bool> requireNoCollInTimeRangeStrict{"requireNoCollInTimeRangeStrict", false, "reject collisions corrupted by the cannibalism, with other collisions within +/- 10 microseconds"};
    Configurable<bool> requireNoCollInTimeRangeNarrow{"requireNoCollInTimeRangeNarrow", false, "reject collisions corrupted by the cannibalism, with other collisions within +/- 2 microseconds"};
    Configurable<bool> requireNoCollInTimeRangeVzDep{"requireNoCollInTimeRangeVzDep", false, "reject collisions corrupted by the cannibalism, with other collisions with pvZ of drifting TPC tracks from past/future collisions within 2.5 cm the current pvZ"};
    Configurable<bool> requireNoCollInROFStd{"requireNoCollInROFStd", false, "reject collisions corrupted by the cannibalism, with other collisions within the same ITS ROF with mult. above a certain threshold"};
    Configurable<bool> requireNoCollInROFStrict{"requireNoCollInROFStrict", false, "reject collisions corrupted by the cannibalism, with other collisions within the same ITS ROF"};
    Configurable<bool> requireINEL0{"requireINEL0", true, "require INEL>0 event selection"};
    Configurable<bool> requireINEL1{"requireINEL1", false, "require INEL>1 event selection"};

    Configurable<float> maxZVtxPosition{"maxZVtxPosition", 10., "max Z vtx position"};

    Configurable<bool> useFT0CbasedOccupancy{"useFT0CbasedOccupancy", false, "Use sum of FT0-C amplitudes for estimating occupancy? (if not, use track-based definition)"};
    // fast check on occupancy
    Configurable<float> minOccupancy{"minOccupancy", -1, "minimum occupancy from neighbouring collisions"};
    Configurable<float> maxOccupancy{"maxOccupancy", -1, "maximum occupancy from neighbouring collisions"};
  } eventSelections;

  struct : ConfigurableGroup {
    // std::string prefix = "v0Selections";
    Configurable<int> v0TypeSelection{"v0Selections.v0TypeSelection", 1, "select on a certain V0 type (leave negative if no selection desired)"};

    // Selection criteria: acceptance
    Configurable<float> daughterEtaCut{"v0Selections.daughterEtaCut", 0.8, "max eta for daughters"};

    // Standard 6 topological criteria
    Configurable<float> v0cospa{"v0Selections.v0cospa", 0.97, "min V0 CosPA"};
    Configurable<float> dcav0dau{"v0Selections.dcav0dau", 1.0, "max DCA V0 Daughters (cm)"};
    Configurable<float> dcav0topv{"v0Selections.dcav0topv", .05, "min DCA V0 to PV (cm)"};
    Configurable<float> dcapostopv{"v0Selections.dcapostopv", .05, "min DCA Pion To PV (cm)"};
    Configurable<float> dcanegtopv{"v0Selections.dcanegtopv", .05, "min DCA Proton To PV (cm)"};
    Configurable<float> v0radius{"v0Selections.v0radius", 1.2, "minimum V0 radius (cm)"};
    Configurable<float> v0radiusMax{"v0Selections.v0radiusMax", 1E5, "maximum V0 radius (cm)"};
    Configurable<float> lifetimeCut{"v0Selections.lifetimeCut", 20, "maximum lifetime (cm)"};

    // invariant mass selection
    Configurable<float> v0MassWindow{"v0Selections.v0MassWindow", 0.008, "#Lambda mass (GeV/#it{c}^{2})"};
    Configurable<float> compMassRejection{"v0Selections.compMassRejection", 0.008, "Competing mass rejection (GeV/#it{c}^{2})"};

    // Additional selection on the AP plot (exclusive for K0Short)
    // original equation: lArmPt*5>TMath::Abs(lArmAlpha)
    Configurable<float> armPodCut{"v0Selections.armPodCut", 5.0f, "pT * (cut) > |alpha|, AP cut. Negative: no cut"};

    // Track quality
    Configurable<int> minTPCrows{"v0Selections.minTPCrows", 70, "minimum TPC crossed rows"};
    Configurable<int> minITSclusters{"v0Selections.minITSclusters", -1, "minimum ITS clusters"};
    Configurable<bool> skipTPConly{"v0Selections.skipTPConly", false, "skip V0s comprised of at least one TPC only prong"};
    Configurable<bool> requirePosITSonly{"v0Selections.requirePosITSonly", false, "require that positive track is ITSonly (overrides TPC quality)"};
    Configurable<bool> requireNegITSonly{"v0Selections.requireNegITSonly", false, "require that negative track is ITSonly (overrides TPC quality)"};
    Configurable<bool> rejectPosITSafterburner{"v0Selections.rejectPosITSafterburner", false, "reject positive track formed out of afterburner ITS tracks"};
    Configurable<bool> rejectNegITSafterburner{"v0Selections.rejectNegITSafterburner", false, "reject negative track formed out of afterburner ITS tracks"};

    // PID (TPC/TOF)
    Configurable<float> tpcPidNsigmaCut{"v0Selections.tpcPidNsigmaCut", 5, "tpcPidNsigmaCut"};
    Configurable<float> tofPidNsigmaCutLaPr{"v0Selections.tofPidNsigmaCutLaPr", 1e+6, "tofPidNsigmaCutLaPr"};
    Configurable<float> tofPidNsigmaCutLaPi{"v0Selections.tofPidNsigmaCutLaPi", 1e+6, "tofPidNsigmaCutLaPi"};
    Configurable<float> tofPidNsigmaCutK0Pi{"v0Selections.tofPidNsigmaCutK0Pi", 1e+6, "tofPidNsigmaCutK0Pi"};

    // PID (TOF)
    Configurable<float> maxDeltaTimeProton{"v0Selections.maxDeltaTimeProton", 1e+9, "check maximum allowed time"};
    Configurable<float> maxDeltaTimePion{"v0Selections.maxDeltaTimePion", 1e+9, "check maximum allowed time"};
  } v0Selections;

  struct : ConfigurableGroup {
    // std::string prefix = "photonSelections";
    Configurable<int> v0TypeSelection{"photonSelections.v0TypeSelection", 1, "select on a certain V0 type (leave negative if no selection desired)"};

    // Selection criteria: acceptance
    Configurable<float> daughterEtaCut{"photonSelections.daughterEtaCut", 0.8, "max eta for daughters"};
    Configurable<float> photonZMax{"photonSelections.photonZMax", 240, "Max photon conversion point z value (cm)"};

    // Standard 6 topological criteria
    Configurable<float> v0cospa{"photonSelections.v0cospa", 0.97, "min V0 CosPA"};
    Configurable<float> dcav0dau{"photonSelections.dcav0dau", 1.0, "max DCA V0 Daughters (cm)"};
    Configurable<float> dcav0topv{"photonSelections.dcav0topv", .05, "min DCA V0 to PV (cm)"};
    Configurable<float> dcanegtopv{"photonSelections.dcanegtopv", .05, "min DCA neg. (e-) To PV (cm)"};
    Configurable<float> dcapostopv{"photonSelections.dcapostopv", .05, "min DCA pos. (e+) To PV (cm)"};
    Configurable<float> v0radius{"photonSelections.v0radius", 1.2, "minimum V0 radius (cm)"};
    Configurable<float> v0radiusMax{"photonSelections.v0radiusMax", 1E5, "maximum V0 radius (cm)"};

    // invariant mass selection
    Configurable<float> photonMassMax{"photonSelections.photonMassMax", 0.008, "#gamma mass (GeV/#it{c}^{2})"};

    // Additional selection on the AP plot
    // original equation: lArmPt*5>TMath::Abs(lArmAlpha)
    Configurable<float> armPodCut{"photonSelections.armPodCut", 5.0f, "pT * (cut) > |alpha|, AP cut. Negative: no cut"};

    // Track quality
    Configurable<int> minTPCrows{"photonSelections.minTPCrows", 70, "minimum TPC crossed rows"};
    Configurable<int> minITSclusters{"photonSelections.minITSclusters", -1, "minimum ITS clusters"};
    Configurable<bool> skipTPConly{"photonSelections.skipTPConly", false, "skip V0s comprised of at least one TPC only prong"};
    Configurable<bool> requirePosITSonly{"photonSelections.requirePosITSonly", false, "require that positive track is ITSonly (overrides TPC quality)"};
    Configurable<bool> requireNegITSonly{"photonSelections.requireNegITSonly", false, "require that negative track is ITSonly (overrides TPC quality)"};
    Configurable<bool> rejectPosITSafterburner{"photonSelections.rejectPosITSafterburner", false, "reject positive track formed out of afterburner ITS tracks"};
    Configurable<bool> rejectNegITSafterburner{"photonSelections.rejectNegITSafterburner", false, "reject negative track formed out of afterburner ITS tracks"};

    // PID (TPC)
    Configurable<float> tpcPidNsigmaCut{"photonSelections.tpcPidNsigmaCut", 5, "tpcPidNsigmaCut"};
  } photonSelections;

  // Machine learning evaluation for pre-selection and corresponding information generation
  o2::ml::OnnxModel mlCustomModelK0Short;
  o2::ml::OnnxModel mlCustomModelLambda;
  o2::ml::OnnxModel mlCustomModelAntiLambda;
  o2::ml::OnnxModel mlCustomModelGamma;

  struct : ConfigurableGroup {
    // ML classifiers: master flags to control whether we should use custom ML classifiers or the scores in the derived data
    Configurable<bool> useK0ShortScores{"mlConfigurations.useK0ShortScores", false, "use ML scores to select K0Short"};
    Configurable<bool> useLambdaScores{"mlConfigurations.useLambdaScores", false, "use ML scores to select Lambda"};
    Configurable<bool> useAntiLambdaScores{"mlConfigurations.useAntiLambdaScores", false, "use ML scores to select AntiLambda"};
    Configurable<bool> useGammaScores{"mlConfigurations.useGammaScores", false, "use ML scores to select Gammas"};

    Configurable<bool> calculateK0ShortScores{"mlConfigurations.calculateK0ShortScores", false, "calculate K0Short ML scores"};
    Configurable<bool> calculateLambdaScores{"mlConfigurations.calculateLambdaScores", false, "calculate Lambda ML scores"};
    Configurable<bool> calculateAntiLambdaScores{"mlConfigurations.calculateAntiLambdaScores", false, "calculate AntiLambda ML scores"};
    Configurable<bool> calculateGammaScores{"mlConfigurations.calculateGammaScores", false, "calculate Gamma ML scores"};

    // ML input for ML calculation
    Configurable<std::string> customModelPathCCDB{"mlConfigurations.customModelPathCCDB", "", "Custom ML Model path in CCDB"};
    Configurable<int64_t> timestampCCDB{"mlConfigurations.timestampCCDB", -1, "timestamp of the ONNX file for ML model used to query in CCDB.  Exceptions: > 0 for the specific timestamp, 0 gets the run dependent timestamp"};
    Configurable<bool> loadCustomModelsFromCCDB{"mlConfigurations.loadCustomModelsFromCCDB", false, "Flag to enable or disable the loading of custom models from CCDB"};
    Configurable<bool> enableOptimizations{"mlConfigurations.enableOptimizations", false, "Enables the ONNX extended model-optimization: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED)"};

    // Local paths for test purposes
    Configurable<std::string> localModelPathLambda{"mlConfigurations.localModelPathLambda", "Lambda_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
    Configurable<std::string> localModelPathAntiLambda{"mlConfigurations.localModelPathAntiLambda", "AntiLambda_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
    Configurable<std::string> localModelPathK0Short{"mlConfigurations.localModelPathK0Short", "KZeroShort_BDTModel.onnx", "(std::string) Path to the local .onnx file."};
    Configurable<std::string> localModelPathGamma{"mlConfigurations.localModelPathGamma", "Gamma_BDTModel.onnx", "(std::string) Path to the local .onnx file."};

    // Thresholds for choosing to populate V0Cores tables with pre-selections
    Configurable<float> thresholdLambda{"mlConfigurations.thresholdLambda", -1.0f, "Threshold to keep Lambda candidates"};
    Configurable<float> thresholdAntiLambda{"mlConfigurations.thresholdAntiLambda", -1.0f, "Threshold to keep AntiLambda candidates"};
    Configurable<float> thresholdK0Short{"mlConfigurations.thresholdK0Short", -1.0f, "Threshold to keep K0Short candidates"};
    Configurable<float> thresholdGamma{"mlConfigurations.thresholdGamma", -1.0f, "Threshold to keep Gamma candidates"};
  } mlConfigurations;

  // CCDB options
  struct : ConfigurableGroup {
    Configurable<std::string> ccdburl{"ccdbConfigurations.ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
    Configurable<std::string> grpPath{"ccdbConfigurations.grpPath", "GLO/GRP/GRP", "Path of the grp file"};
    Configurable<std::string> grpmagPath{"ccdbConfigurations.grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
    Configurable<std::string> lutPath{"ccdbConfigurations.lutPath", "GLO/Param/MatLUT", "Path of the Lut parametrization"};
    Configurable<std::string> geoPath{"ccdbConfigurations.geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
    Configurable<std::string> mVtxPath{"ccdbConfigurations.mVtxPath", "GLO/Calib/MeanVertex", "Path of the mean vertex file"};
  } ccdbConfigurations;

  Service<o2::ccdb::BasicCCDBManager> ccdb;
  o2::ccdb::CcdbApi ccdbApi;
  int mRunNumber;
  std::map<std::string, std::string> metadata;

  Zorro zorro;
  OutputObj<ZorroSummary> zorroSummary{"zorroSummary"};

  ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f, 1.8f, 2.0f, 2.4f, 2.8f, 3.2f, 3.6f, 4.0f, 4.8f, 5.6f, 6.5f, 7.5f, 9.0f, 11.0f, 13.0f, 15.0f, 19.0f, 23.0f, 30.0f, 40.0f, 50.0f}, "pt axis for analysis"};
  ConfigurableAxis axisResonanceMass{"axisResonanceMass", {550, 0.450f, 1.000f}, "M (K^{0}_{S} #gamma) (GeV/#it{c}^{2})"};
  ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f}, "Centrality"};
  ConfigurableAxis axisNch{"axisNch", {500, 0.0f, +5000.0f}, "Number of charged particles"};

  ConfigurableAxis axisRawCentrality{"axisRawCentrality", {VARIABLE_WIDTH, 0.000f, 52.320f, 75.400f, 95.719f, 115.364f, 135.211f, 155.791f, 177.504f, 200.686f, 225.641f, 252.645f, 281.906f, 313.850f, 348.302f, 385.732f, 426.307f, 470.146f, 517.555f, 568.899f, 624.177f, 684.021f, 748.734f, 818.078f, 892.577f, 973.087f, 1058.789f, 1150.915f, 1249.319f, 1354.279f, 1465.979f, 1584.790f, 1710.778f, 1844.863f, 1985.746f, 2134.643f, 2291.610f, 2456.943f, 2630.653f, 2813.959f, 3006.631f, 3207.229f, 3417.641f, 3637.318f, 3865.785f, 4104.997f, 4354.938f, 4615.786f, 4885.335f, 5166.555f, 5458.021f, 5762.584f, 6077.881f, 6406.834f, 6746.435f, 7097.958f, 7462.579f, 7839.165f, 8231.629f, 8635.640f, 9052.000f, 9484.268f, 9929.111f, 10389.350f, 10862.059f, 11352.185f, 11856.823f, 12380.371f, 12920.401f, 13476.971f, 14053.087f, 14646.190f, 15258.426f, 15890.617f, 16544.433f, 17218.024f, 17913.465f, 18631.374f, 19374.983f, 20136.700f, 20927.783f, 21746.796f, 22590.880f, 23465.734f, 24372.274f, 25314.351f, 26290.488f, 27300.899f, 28347.512f, 29436.133f, 30567.840f, 31746.818f, 32982.664f, 34276.329f, 35624.859f, 37042.588f, 38546.609f, 40139.742f, 41837.980f, 43679.429f, 45892.130f, 400000.000f}, "raw centrality signal"}; // for QA

  ConfigurableAxis axisOccupancy{"axisOccupancy", {VARIABLE_WIDTH, 0.0f, 250.0f, 500.0f, 750.0f, 1000.0f, 1500.0f, 2000.0f, 3000.0f, 4500.0f, 6000.0f, 8000.0f, 10000.0f, 50000.0f}, "Occupancy"};

  // topological variable QA axes
  ConfigurableAxis axisDCAtoPV{"axisDCAtoPV", {20, 0.0f, 1.0f}, "DCA (cm)"};
  ConfigurableAxis axisDCAdau{"axisDCAdau", {20, 0.0f, 2.0f}, "DCA (cm)"};
  ConfigurableAxis axisDCAV0ToPV{"axisDCAV0ToPV", {20, 0.0f, 2.0f}, "DCA (cm)"};
  ConfigurableAxis axisPointingAngle{"axisPointingAngle", {20, 0.0f, 2.0f}, "pointing angle (rad)"};
  ConfigurableAxis axisRadius{"axisRadius", {20, 0.0f, 60.0f}, "Decay radius (cm)"};
  ConfigurableAxis axisProperLifeTime{"axisV0ProperLifeTime", {100, 0.0f, 50.0f}, "ProperLifeTime 2D radius (cm)"};
  ConfigurableAxis axisMassWindow{"axisMassWindow", {40, -0.020f, 0.020f}, "Inv. mass - PDG mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisPhotonMass{"axisPhotonMass", {500, 0.0f, 0.50f}, "Photon inv. mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisPhotonZconv{"axisPhotonZconv", {500, 0.0f, 500.0f}, "Max photon conversion point z value (cm)"};
  ConfigurableAxis axisK0Mass{"axisK0Mass", {500, 0.400f, 0.600f}, "K0Short mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisLambdaMass{"axisLambdaMass", {500, 1.098f, 1.198f}, "Lambda mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisXiMass{"axisXiMass", {500, 1.318f, 1.370f}, "Xi mass (GeV/#it{c}^{2})"};
  ConfigurableAxis axisNsigmaTPC{"axisNsigmaTPC", {200, -10.0f, 10.0f}, "N sigma TPC"};

  // AP plot axes
  ConfigurableAxis axisAPAlpha{"axisAPAlpha", {220, -1.1f, 1.1f}, "V0 AP alpha"};
  ConfigurableAxis axisAPQt{"axisAPQt", {220, 0.0f, 0.5f}, "V0 AP alpha"};

  // Track quality axes
  ConfigurableAxis axisTPCrows{"axisTPCrows", {160, 0.0f, 160.0f}, "N TPC rows"};
  ConfigurableAxis axisITSclus{"axisITSclus", {7, 0.0f, 7.0f}, "N ITS Clusters"};

  // UPC axes
  ConfigurableAxis axisSelGap{"axisSelGap", {4, -1.5, 2.5}, "Gap side"};

  // PDG database
  Service<o2::framework::O2DatabasePDG> pdgDB;

  // For manual sliceBy
  PresliceUnsorted<soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraCollLabels>> perMcCollision = aod::v0data::straMCCollisionId;

  void init(InitContext const&)
  {
    // Event Counters
    histos.add("hEventSelection", "hEventSelection", kTH1F, {{20, -0.5f, +19.5f}});
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(1, "All collisions");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(2, "sel8 cut");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(3, "kIsTriggerTVX");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(4, "kNoITSROFrameBorder");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(5, "kNoTimeFrameBorder");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(6, "posZ cut");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(7, "kIsVertexITSTPC");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(8, "kIsGoodZvtxFT0vsPV");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(9, "kIsVertexTOFmatched");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(10, "kIsVertexTRDmatched");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(11, "kNoSameBunchPileup");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(12, "kNoCollInTimeRangeStd");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(13, "kNoCollInTimeRangeStrict");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(14, "kNoCollInTimeRangeNarrow");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(15, "kNoCollInTimeRangeVzDep");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(16, "kNoCollInRofStd");
    histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(17, "kNoCollInRofStrict");
    if (doPPAnalysis) {
      histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(18, "INEL>0");
      histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(19, "INEL>1");
    } else {
      histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(18, "Below min occup.");
      histos.get<TH1>(HIST("hEventSelection"))->GetXaxis()->SetBinLabel(19, "Above max occup.");
    }

    histos.add("hEventCentrality", "hEventCentrality", kTH1F, {{100, 0.0f, +100.0f}});
    histos.add("hCentralityVsNch", "hCentralityVsNch", kTH2F, {axisCentrality, axisNch});

    histos.add("hEventPVz", "hEventPVz", kTH1F, {{100, -20.0f, +20.0f}});
    histos.add("hCentralityVsPVz", "hCentralityVsPVz", kTH2F, {axisCentrality, {100, -20.0f, +20.0f}});

    histos.add("hEventOccupancy", "hEventOccupancy", kTH1F, {axisOccupancy});
    histos.add("hCentralityVsOccupancy", "hCentralityVsOccupancy", kTH2F, {axisCentrality, axisOccupancy});

    if (!doPPAnalysis) {
      histos.add("hGapSide", "Gap side; Entries", kTH1F, {{5, -0.5, 4.5}});
      histos.add("hSelGapSide", "Selected gap side; Entries", kTH1F, {axisSelGap});
      histos.add("hEventCentralityVsSelGapSide", ";Centrality (%); Selected gap side", kTH2F, {{100, 0.0f, +100.0f}, axisSelGap});
    }

    // for QA and test purposes
    auto hRawCentrality = histos.add<TH1>("hRawCentrality", "hRawCentrality", kTH1F, {axisRawCentrality});

    for (int ii = 1; ii < 101; ii++) {
      float value = 100.5f - static_cast<float>(ii);
      hRawCentrality->SetBinContent(ii, value);
    }

    // histograms versus mass
    histos.add("K0sGamma/h3dMassK0sGamma", "h3dMassK0sGamma", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
    if (!doPPAnalysis) {
      // Non-UPC info
      histos.add("K0sGamma/h3dMassK0sGammaHadronic", "h3dMassK0sGammaHadronic", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
      // UPC info
      histos.add("K0sGamma/h3dMassK0sGammaSGA", "h3dMassK0sGammaSGA", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
      histos.add("K0sGamma/h3dMassK0sGammaSGC", "h3dMassK0sGammaSGC", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
      histos.add("K0sGamma/h3dMassK0sGammaDG", "h3dMassK0sGammaDG", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
    }
    histos.add("K0sGamma/h2dNbrOfK0ShortVsCentrality", "h2dNbrOfK0ShortVsCentrality", kTH2F, {axisCentrality, {10, -0.5f, 9.5f}});
    histos.add("K0sGamma/h2dNbrOfGammaVsCentrality", "h2dNbrOfGammaVsCentrality", kTH2F, {axisCentrality, {10, -0.5f, 9.5f}});
    // QA plot
    // Candidates before selections
    histos.add("K0sGamma/BeforeSel/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/BeforeSel/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/BeforeSel/hDCAV0Daughters", "hDCAV0Daughters", kTH1F, {axisDCAdau});
    histos.add("K0sGamma/BeforeSel/hDCAV0ToPV", "hDCAV0ToPV", kTH1F, {axisDCAV0ToPV});
    histos.add("K0sGamma/BeforeSel/hV0PointingAngle", "hV0PointingAngle", kTH1F, {axisPointingAngle});
    histos.add("K0sGamma/BeforeSel/hV0Radius", "hV0Radius", kTH1F, {axisRadius});
    histos.add("K0sGamma/BeforeSel/hV0DecayLength", "hDecayLength", kTH1F, {axisProperLifeTime});
    histos.add("K0sGamma/BeforeSel/hV0InvMassWindow", "hInvMassWindow", kTH1F, {axisMassWindow});
    histos.add("K0sGamma/BeforeSel/h2dCompetingMassRej", "h2dCompetingMassRej", kTH2F, {axisLambdaMass, axisK0Mass});
    histos.add("K0sGamma/BeforeSel/hPhotonMass", "hPhotonMass", kTH1F, {axisPhotonMass});
    histos.add("K0sGamma/BeforeSel/hPhotonZconv", "hPhotonZconv", kTH1F, {axisPhotonZconv});
    histos.add("K0sGamma/BeforeSel/h2dArmenteros", "h2dArmenteros", kTH2F, {axisAPAlpha, axisAPQt});
    histos.add("K0sGamma/BeforeSel/hPosTPCNsigmaPi", "hPosTPCNsigmaPi", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/BeforeSel/hNegTPCNsigmaPi", "hNegTPCNsigmaPi", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/BeforeSel/hPosTPCNsigmaEl", "hPosTPCNsigmaEl", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/BeforeSel/hNegTPCNsigmaEl", "hNegTPCNsigmaEl", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/BeforeSel/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    histos.add("K0sGamma/BeforeSel/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    // Candidates after K0s selections
    histos.add("K0sGamma/K0s/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/K0s/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/K0s/hDCAV0Daughters", "hDCAV0Daughters", kTH1F, {axisDCAdau});
    histos.add("K0sGamma/K0s/hDCAV0ToPV", "hDCAV0ToPV", kTH1F, {axisDCAV0ToPV});
    histos.add("K0sGamma/K0s/hV0PointingAngle", "hV0PointingAngle", kTH1F, {axisPointingAngle});
    histos.add("K0sGamma/K0s/hV0Radius", "hV0Radius", kTH1F, {axisRadius});
    histos.add("K0sGamma/K0s/hV0DecayLength", "hDecayLength", kTH1F, {axisProperLifeTime});
    histos.add("K0sGamma/K0s/hV0InvMassWindow", "hInvMassWindow", kTH1F, {axisMassWindow});
    histos.add("K0sGamma/K0s/h2dCompetingMassRej", "h2dCompetingMassRej", kTH2F, {axisLambdaMass, axisK0Mass});
    histos.add("K0sGamma/K0s/h2dArmenteros", "h2dArmenteros", kTH2F, {axisAPAlpha, axisAPQt});
    histos.add("K0sGamma/K0s/hPosTPCNsigma", "hPosTPCNsigma", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/K0s/hNegTPCNsigma", "hNegTPCNsigma", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/K0s/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    histos.add("K0sGamma/K0s/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    // Candidates after Gamma selections
    histos.add("K0sGamma/Gamma/hPosDCAToPV", "hPosDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/Gamma/hNegDCAToPV", "hNegDCAToPV", kTH1F, {axisDCAtoPV});
    histos.add("K0sGamma/Gamma/hDCAV0Daughters", "hDCAV0Daughters", kTH1F, {axisDCAdau});
    histos.add("K0sGamma/Gamma/hDCAV0ToPV", "hDCAV0ToPV", kTH1F, {axisDCAV0ToPV});
    histos.add("K0sGamma/Gamma/hV0PointingAngle", "hV0PointingAngle", kTH1F, {axisPointingAngle});
    histos.add("K0sGamma/Gamma/hV0Radius", "hV0Radius", kTH1F, {axisRadius});
    histos.add("K0sGamma/Gamma/hPhotonMass", "hPhotonMass", kTH1F, {axisPhotonMass});
    histos.add("K0sGamma/Gamma/hPhotonZconv", "hPhotonZconv", kTH1F, {axisPhotonZconv});
    histos.add("K0sGamma/Gamma/h2dArmenteros", "h2dArmenteros", kTH2F, {axisAPAlpha, axisAPQt});
    histos.add("K0sGamma/Gamma/hPosTPCNsigma", "hPosTPCNsigma", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/Gamma/hNegTPCNsigma", "hNegTPCNsigma", kTH1F, {axisNsigmaTPC});
    histos.add("K0sGamma/Gamma/h2dPositiveITSvsTPCpts", "h2dPositiveITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    histos.add("K0sGamma/Gamma/h2dNegativeITSvsTPCpts", "h2dNegativeITSvsTPCpts", kTH2F, {axisTPCrows, axisITSclus});
    if (doMCAssociation) {
      histos.add("K0sGamma/h3dInvMassTrueK0Star892", "h3dInvMassTrueK0Star892", kTH3F, {axisCentrality, axisPt, axisResonanceMass});
    }

    if (cfgSkimmedProcessing) {
      zorroSummary.setObject(zorro.getZorroSummary());
    }

    // inspect histogram sizes, please
    histos.print();
  }

  template <typename TCollision> // TCollision should be of the type: soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraStamps>::iterator or so
  void initCCDB(TCollision const& collision)
  {
    if (mRunNumber == collision.runNumber()) {
      return;
    }

    mRunNumber = collision.runNumber();
    if (cfgSkimmedProcessing) {
      ccdb->setURL(ccdbConfigurations.ccdburl);
      ccdb->setCaching(true);
      ccdb->setLocalObjectValidityChecking();
      ccdb->setFatalWhenNull(false);

      zorro.initCCDB(ccdb.service, collision.runNumber(), collision.timestamp(), cfgSkimmedTrigger.value);
      zorro.populateHistRegistry(histos, collision.runNumber());
    }

    // machine learning initialization if requested
    if (mlConfigurations.calculateK0ShortScores ||
        mlConfigurations.calculateLambdaScores ||
        mlConfigurations.calculateAntiLambdaScores ||
        mlConfigurations.calculateGammaScores) {
      int64_t timeStampML = collision.timestamp();
      if (mlConfigurations.timestampCCDB.value != -1)
        timeStampML = mlConfigurations.timestampCCDB.value;
      loadMachines(timeStampML);
    }
  }

  // function to load models for ML-based classifiers
  void loadMachines(int64_t timeStampML)
  {
    if (mlConfigurations.loadCustomModelsFromCCDB) {
      ccdbApi.init(ccdbConfigurations.ccdburl);
      LOG(info) << "Fetching models for timestamp: " << timeStampML;

      if (mlConfigurations.calculateLambdaScores) {
        bool retrieveSuccessLambda = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathLambda.value);
        if (retrieveSuccessLambda) {
          mlCustomModelLambda.initModel(mlConfigurations.localModelPathLambda.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the Lambda model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }

      if (mlConfigurations.calculateAntiLambdaScores) {
        bool retrieveSuccessAntiLambda = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathAntiLambda.value);
        if (retrieveSuccessAntiLambda) {
          mlCustomModelAntiLambda.initModel(mlConfigurations.localModelPathAntiLambda.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the AntiLambda model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }

      if (mlConfigurations.calculateK0ShortScores) {
        bool retrieveSuccessKZeroShort = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathK0Short.value);
        if (retrieveSuccessKZeroShort) {
          mlCustomModelK0Short.initModel(mlConfigurations.localModelPathK0Short.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the K0Short model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }

      if (mlConfigurations.calculateGammaScores) {
        bool retrieveSuccessGamma = ccdbApi.retrieveBlob(mlConfigurations.customModelPathCCDB, ".", metadata, timeStampML, false, mlConfigurations.localModelPathGamma.value);
        if (retrieveSuccessGamma) {
          mlCustomModelGamma.initModel(mlConfigurations.localModelPathGamma.value, mlConfigurations.enableOptimizations.value);
        } else {
          LOG(fatal) << "Error encountered while fetching/loading the Gamma model from CCDB! Maybe the model doesn't exist yet for this runnumber/timestamp?";
        }
      }
    } else {
      if (mlConfigurations.calculateLambdaScores)
        mlCustomModelLambda.initModel(mlConfigurations.localModelPathLambda.value, mlConfigurations.enableOptimizations.value);
      if (mlConfigurations.calculateAntiLambdaScores)
        mlCustomModelAntiLambda.initModel(mlConfigurations.localModelPathAntiLambda.value, mlConfigurations.enableOptimizations.value);
      if (mlConfigurations.calculateK0ShortScores)
        mlCustomModelK0Short.initModel(mlConfigurations.localModelPathK0Short.value, mlConfigurations.enableOptimizations.value);
      if (mlConfigurations.calculateGammaScores)
        mlCustomModelGamma.initModel(mlConfigurations.localModelPathGamma.value, mlConfigurations.enableOptimizations.value);
    }
    LOG(info) << "ML Models loaded.";
  }

  template <typename TCollision>
  bool isEventAccepted(TCollision collision, bool fillHists)
  // check whether the collision passes our collision selections
  {
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 0. /* all collisions */);

    if (eventSelections.requireSel8 && !collision.sel8()) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 1 /* sel8 collisions */);

    if (eventSelections.requireTriggerTVX && !collision.selection_bit(aod::evsel::kIsTriggerTVX)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 2 /* FT0 vertex (acceptable FT0C-FT0A time difference) collisions */);

    if (eventSelections.rejectITSROFBorder && !collision.selection_bit(o2::aod::evsel::kNoITSROFrameBorder)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 3 /* Not at ITS ROF border */);

    if (eventSelections.rejectTFBorder && !collision.selection_bit(o2::aod::evsel::kNoTimeFrameBorder)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 4 /* Not at TF border */);

    if (std::abs(collision.posZ()) > eventSelections.maxZVtxPosition) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 5 /* vertex-Z selected */);

    if (eventSelections.requireIsVertexITSTPC && !collision.selection_bit(o2::aod::evsel::kIsVertexITSTPC)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 6 /* Contains at least one ITS-TPC track */);

    if (eventSelections.requireIsGoodZvtxFT0VsPV && !collision.selection_bit(o2::aod::evsel::kIsGoodZvtxFT0vsPV)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 7 /* PV position consistency check */);

    if (eventSelections.requireIsVertexTOFmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTOFmatched)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 8 /* PV with at least one contributor matched with TOF */);

    if (eventSelections.requireIsVertexTRDmatched && !collision.selection_bit(o2::aod::evsel::kIsVertexTRDmatched)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 9 /* PV with at least one contributor matched with TRD */);

    if (eventSelections.rejectSameBunchPileup && !collision.selection_bit(o2::aod::evsel::kNoSameBunchPileup)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 10 /* Not at same bunch pile-up */);

    if (eventSelections.requireNoCollInTimeRangeStd && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeStandard)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 11 /* No other collision within +/- 2 microseconds or mult above a certain threshold in -4 - -2 microseconds*/);

    if (eventSelections.requireNoCollInTimeRangeStrict && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeStrict)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 12 /* No other collision within +/- 10 microseconds */);

    if (eventSelections.requireNoCollInTimeRangeNarrow && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeNarrow)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 13 /* No other collision within +/- 2 microseconds */);

    if (eventSelections.requireNoCollInTimeRangeVzDep && !collision.selection_bit(o2::aod::evsel::kNoCollInTimeRangeVzDependent)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 14 /* No other collision with pvZ of drifting TPC tracks from past/future collisions within 2.5 cm the current pvZ */);

    if (eventSelections.requireNoCollInROFStd && !collision.selection_bit(o2::aod::evsel::kNoCollInRofStandard)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 15 /* No other collision within the same ITS ROF with mult. above a certain threshold */);

    if (eventSelections.requireNoCollInROFStrict && !collision.selection_bit(o2::aod::evsel::kNoCollInRofStrict)) {
      return false;
    }
    if (fillHists)
      histos.fill(HIST("hEventSelection"), 16 /* No other collision within the same ITS ROF */);

    if (doPPAnalysis) { // we are in pp
      if (eventSelections.requireINEL0 && collision.multNTracksPVeta1() < 1) {
        return false;
      }
      if (fillHists)
        histos.fill(HIST("hEventSelection"), 17 /* INEL > 0 */);

      if (eventSelections.requireINEL1 && collision.multNTracksPVeta1() < 2) {
        return false;
      }
      if (fillHists)
        histos.fill(HIST("hEventSelection"), 18 /* INEL > 1 */);

    } else { // we are in Pb-Pb
      float collisionOccupancy = eventSelections.useFT0CbasedOccupancy ? collision.ft0cOccupancyInTimeRange() : collision.trackOccupancyInTimeRange();
      if (eventSelections.minOccupancy >= 0 && collisionOccupancy < eventSelections.minOccupancy) {
        return false;
      }
      if (fillHists)
        histos.fill(HIST("hEventSelection"), 17 /* Below min occupancy */);

      if (eventSelections.maxOccupancy >= 0 && collisionOccupancy > eventSelections.maxOccupancy) {
        return false;
      }
      if (fillHists)
        histos.fill(HIST("hEventSelection"), 18 /* Above max occupancy */);
    }

    return true;
  }

  template <typename TCollision>
  void fillEventHistograms(TCollision collision, float& centrality, int& selGapSide)
  {
    if (doPPAnalysis) { //
      centrality = collision.centFT0M();

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0A() + collision.multFT0C()));
      }
    } else {
      centrality = collision.centFT0C();

      if (qaCentrality) {
        auto hRawCentrality = histos.get<TH1>(HIST("hRawCentrality"));
        centrality = hRawCentrality->GetBinContent(hRawCentrality->FindBin(collision.multFT0C()));
      }
    }

    // in case we want to push the analysis to Pb-Pb UPC
    int gapSide = collision.gapSide();
    if (!doPPAnalysis) {
      // -1 --> Hadronic
      // 0 --> Single Gap - A side
      // 1 --> Single Gap - C side
      // 2 --> Double Gap - both A & C sides
      selGapSide = sgSelector.trueGap(collision, upcCuts.fv0Cut, upcCuts.ft0aCut, upcCuts.ft0cCut, upcCuts.zdcCut);
      histos.fill(HIST("hGapSide"), gapSide);
      histos.fill(HIST("hSelGapSide"), selGapSide);
      histos.fill(HIST("hEventCentralityVsSelGapSide"), centrality, selGapSide <= 2 ? selGapSide : -1);
    }

    histos.fill(HIST("hEventCentrality"), centrality);

    histos.fill(HIST("hCentralityVsNch"), centrality, collision.multNTracksPVeta1());

    histos.fill(HIST("hCentralityVsPVz"), centrality, collision.posZ());
    histos.fill(HIST("hEventPVz"), collision.posZ());

    histos.fill(HIST("hEventOccupancy"), collision.trackOccupancyInTimeRange());
    histos.fill(HIST("hCentralityVsOccupancy"), centrality, collision.trackOccupancyInTimeRange());

    return;
  }

  template <typename TV0, typename TCollision>
  uint64_t isV0Selected(TV0 v0, TCollision collision, bool isPhoton)
  // precalculate this information so that a check is one mask operation, not many
  {
    if (isPhoton) {
      //
      // Acceptance variables
      //
      if (v0.z() > photonSelections.photonZMax)
        return false;

      if (std::abs(v0.negativeeta()) > photonSelections.daughterEtaCut || std::abs(v0.positiveeta()) > photonSelections.daughterEtaCut)
        return false; // remove acceptance that's badly reproduced by MC / superfluous in future

      if (photonSelections.v0TypeSelection > -1 && v0.v0Type() != photonSelections.v0TypeSelection)
        return false; // skip V0s that are not standard

      //
      // Base topological variables
      //

      // v0 radius min/max selections
      if (v0.v0radius() < photonSelections.v0radius)
        return false;
      if (v0.v0radius() > photonSelections.v0radiusMax)
        return false;
      // DCA pos and neg to PV
      if (std::fabs(v0.dcapostopv()) < photonSelections.dcapostopv)
        return false;
      if (std::fabs(v0.dcanegtopv()) > photonSelections.dcanegtopv)
        return false;
        
      // V0 cosine of pointing angle
      if (v0.v0cosPA() < photonSelections.v0cospa)
        return false;
      // DCA between v0 daughters
      if (v0.dcaV0daughters() > photonSelections.dcav0dau)
        return false;
      // DCA V0 to prim vtx
      if (v0.dcav0topv() < photonSelections.dcav0topv)
        return false;

      //
      // invariant mass selection
      //
      if (v0.mGamma() > photonSelections.photonMassMax)
        return false;

      auto posTrackExtra = v0.template posTrackExtra_as<DauTracks>();
      auto negTrackExtra = v0.template negTrackExtra_as<DauTracks>();

      //
      // ITS quality flags
      //
      bool posIsFromAfterburner = posTrackExtra.itsChi2PerNcl() < 0;
      bool negIsFromAfterburner = negTrackExtra.itsChi2PerNcl() < 0;
      
      if (posTrackExtra.itsNCls() < photonSelections.minITSclusters)
        return false;
      if (negTrackExtra.itsNCls() < photonSelections.minITSclusters)
        return false;
      if (photonSelections.rejectPosITSafterburner && posIsFromAfterburner)
        return false;
      if (photonSelections.rejectNegITSafterburner && negIsFromAfterburner)
        return false;

      //
      // TPC quality flags
      //
      if (posTrackExtra.tpcCrossedRows() < photonSelections.minTPCrows)
        return false;
      if (negTrackExtra.tpcCrossedRows() < photonSelections.minTPCrows)
        return false;

      //
      // TPC PID
      //
      if (std::fabs(posTrackExtra.tpcNSigmaEl()) > photonSelections.tpcPidNsigmaCut)
        return false;
      if (std::fabs(negTrackExtra.tpcNSigmaEl()) > photonSelections.tpcPidNsigmaCut)
        return false;

      //
      // ITS only tag
      if (photonSelections.requirePosITSonly && posTrackExtra.tpcCrossedRows() > 0)
        return false;
      if (photonSelections.requireNegITSonly && negTrackExtra.tpcCrossedRows() > 0)
        return false;

      //
      // TPC only tag
      if (photonSelections.skipTPConly && posTrackExtra.detectorMap() == o2::aod::track::TPC)
        return false;
      if (photonSelections.skipTPConly && negTrackExtra.detectorMap() == o2::aod::track::TPC)
        return false;

      //
      // armenteros
      if (photonSelections.armPodCut > 1e-4 && v0.qtarm() * photonSelections.armPodCut < std::fabs(v0.alpha()))
        return false;

      //
      // MC association (if asked)
      if (doMCAssociation) {
        if constexpr (requires { v0.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>(); }) { // check if MC information is available
          auto v0MC = v0.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

          if (v0MC.pdgCode() != 22 || v0MC.pdgCodePositive() != 11 || v0MC.pdgCodeNegative() != -11)
            return false;
        }
      }

    } else {
      //
      // Acceptance variables
      //
      if (std::abs(v0.negativeeta()) > photonSelections.daughterEtaCut || std::abs(v0.positiveeta()) > photonSelections.daughterEtaCut)
        return false; // remove acceptance that's badly reproduced by MC / superfluous in future

      if (photonSelections.v0TypeSelection > -1 && v0.v0Type() != photonSelections.v0TypeSelection)
        return false; // skip V0s that are not standard

      //
      // Base topological variables
      //

      // v0 radius min/max selections
      if (v0.v0radius() < v0Selections.v0radius)
        return false;
      if (v0.v0radius() > v0Selections.v0radiusMax)
        return false;
      // DCA pos and neg to PV
      if (std::fabs(v0.dcapostopv()) < v0Selections.dcapostopv)
        return false;
      if (std::fabs(v0.dcanegtopv()) > v0Selections.dcanegtopv)
        return false;
        
      // V0 cosine of pointing angle
      if (v0.v0cosPA() < v0Selections.v0cospa)
        return false;
      // DCA between v0 daughters
      if (v0.dcaV0daughters() > v0Selections.dcav0dau)
        return false;
      // DCA V0 to prim vtx
      if (v0.dcav0topv() < v0Selections.dcav0topv)
        return false;

      //
      // invariant mass window
      //
      if (std::fabs(v0.mK0Short() - o2::constants::physics::MassK0Short) > v0Selections.v0MassWindow)
        return false;

      //
      // competing mass rejection
      //
      if (std::fabs(v0.mLambda() - o2::constants::physics::MassLambda0) < v0Selections.compMassRejection)
        return false;

      auto posTrackExtra = v0.template posTrackExtra_as<DauTracks>();
      auto negTrackExtra = v0.template negTrackExtra_as<DauTracks>();

      //
      // ITS quality flags
      //
      bool posIsFromAfterburner = posTrackExtra.itsChi2PerNcl() < 0;
      bool negIsFromAfterburner = negTrackExtra.itsChi2PerNcl() < 0;

      if (posTrackExtra.itsNCls() < v0Selections.minITSclusters)
        return false;
      if (negTrackExtra.itsNCls() < v0Selections.minITSclusters)
        return false;
      if (v0Selections.rejectPosITSafterburner && posIsFromAfterburner)
        return false;
      if (v0Selections.rejectNegITSafterburner && negIsFromAfterburner)
        return false;

      //
      // TPC quality flags
      //
      if (posTrackExtra.tpcCrossedRows() < v0Selections.minTPCrows)
        return false;
      if (negTrackExtra.tpcCrossedRows() < v0Selections.minTPCrows)
        return false;

      //
      // TPC PID
      //
      if (std::fabs(posTrackExtra.tpcNSigmaPi()) > v0Selections.tpcPidNsigmaCut)
        return false;
      if (std::fabs(negTrackExtra.tpcNSigmaPi()) > v0Selections.tpcPidNsigmaCut)
        return false;

      //
      // TOF PID in DeltaT
      // Positive track
      if (std::fabs(v0.posTOFDeltaTK0Pi()) > v0Selections.maxDeltaTimePion)
        return false;
      // Negative track
      if (std::fabs(v0.negTOFDeltaTK0Pi()) > v0Selections.maxDeltaTimePion)
        return false;

      //
      // TOF PID in NSigma
      // Positive track
      if (std::fabs(v0.tofNSigmaK0PiPlus()) > v0Selections.tofPidNsigmaCutK0Pi)
        return false;
      // Negative track
      if (std::fabs(v0.tofNSigmaK0PiMinus()) > v0Selections.tofPidNsigmaCutK0Pi)
        return false;

      //
      // ITS only tag
      if (v0Selections.requirePosITSonly && posTrackExtra.tpcCrossedRows() > 0)
        return false;
      if (v0Selections.requireNegITSonly && negTrackExtra.tpcCrossedRows() > 0)
        return false;

      //
      // TPC only tag
      if (v0Selections.skipTPConly && posTrackExtra.detectorMap() == o2::aod::track::TPC)
        return false;
      if (v0Selections.skipTPConly && negTrackExtra.detectorMap() == o2::aod::track::TPC)
        return false;

      //
      // proper lifetime
      if (v0.distovertotmom(collision.posX(), collision.posY(), collision.posZ()) * o2::constants::physics::MassK0Short > v0Selections.lifetimeCut)
        return false;

      //
      // armenteros
      if (v0Selections.armPodCut > 1e-4 && v0.qtarm() * v0Selections.armPodCut < std::fabs(v0.alpha()))
        return false;

      //
      // MC association (if asked)
      if (doMCAssociation) {
        if constexpr (requires { v0.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>(); }) { // check if MC information is available
          auto v0MC = v0.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

          if (v0MC.pdgCode() != 310 || v0MC.pdgCodePositive() != 211 || v0MC.pdgCodeNegative() != -211)
            return false;
        }
      }
    }

    return true;
  }

  template <typename TV0, typename TCollision>
  void analyseV0Candidate(TV0 v0, TCollision collision, float pt, std::vector<bool>& selK0ShortIndices, std::vector<bool>& selGammaIndices, int v0TableOffset)
  // precalculate this information so that a check is one mask operation, not many
  {
    bool passK0ShortSelections = false;
    bool passGammaSelections = false;

    // machine learning is on, go for calculation of thresholds
    // FIXME THIS NEEDS ADJUSTING
    std::vector<float> inputFeatures{pt, 0.0f, 0.0f, v0.v0radius(), v0.v0cosPA(), v0.dcaV0daughters(), v0.dcapostopv(), v0.dcanegtopv()};

    if (mlConfigurations.useK0ShortScores) {
      float k0shortScore = -1;
      if (mlConfigurations.calculateK0ShortScores) {
        // evaluate machine-learning scores
        float* k0shortProbability = mlCustomModelK0Short.evalModel(inputFeatures);
        k0shortScore = k0shortProbability[1];
      } else {
        k0shortScore = v0.k0ShortBDTScore();
      }
      if (k0shortScore > mlConfigurations.thresholdK0Short.value) {
        passK0ShortSelections = true;
      }
    } else {
      passK0ShortSelections = isV0Selected(v0, collision, false);
    }
    if (mlConfigurations.useGammaScores) {
      float gammaScore = -1;
      if (mlConfigurations.calculateGammaScores) {
        float* gammaProbability = mlCustomModelGamma.evalModel(inputFeatures);
        gammaScore = gammaProbability[1];
      }  else {
        gammaScore = v0.antiLambdaBDTScore();
      }
      if (gammaScore > mlConfigurations.thresholdGamma.value) {
        passGammaSelections = true;
      }
    } else {
      passGammaSelections = isV0Selected(v0, collision, true);
    }

    // need local index because of the grouping of collisions
    selK0ShortIndices[v0.globalIndex() - v0TableOffset] = passK0ShortSelections;
    selGammaIndices[v0.globalIndex() - v0TableOffset] = passGammaSelections;
  }

  template <typename TCollision, typename TV0>
  void fillQAplot(TCollision collision, TV0 k0short, TV0 gamma, int afterSel /* : 0 (before selections) ; : 1 (after selections) */)
  { // fill QA information about hyperon - antihyperon pair
    auto posTrackExtraK0Short = k0short.template posTrackExtra_as<DauTracks>();
    auto negTrackExtraK0Short = k0short.template negTrackExtra_as<DauTracks>();

    auto posTrackExtraGamma = gamma.template posTrackExtra_as<DauTracks>();
    auto negTrackExtraGamma = gamma.template negTrackExtra_as<DauTracks>();

    float k0shortDecayLength = std::sqrt(std::pow(k0short.x() - collision.posX(), 2) + std::pow(k0short.y() - collision.posY(), 2) + std::pow(k0short.z() - collision.posZ(), 2)) * o2::constants::physics::MassKaonNeutral / (k0short.p() + 1E-10);
    
    if (afterSel == 0) {
      // Candidates before any selections
      histos.fill(HIST("K0sGamma/BeforeSel/hPosDCAToPV"), k0short.dcapostopv());
      histos.fill(HIST("K0sGamma/BeforeSel/hNegDCAToPV"), k0short.dcanegtopv());
      histos.fill(HIST("K0sGamma/BeforeSel/hDCAV0Daughters"), k0short.dcaV0daughters());
      histos.fill(HIST("K0sGamma/BeforeSel/hDCAV0ToPV"), k0short.dcav0topv());
      histos.fill(HIST("K0sGamma/BeforeSel/hV0PointingAngle"), k0short.v0cosPA());
      histos.fill(HIST("K0sGamma/BeforeSel/hV0Radius"), k0short.v0radius());
      histos.fill(HIST("K0sGamma/BeforeSel/hV0DecayLength"), k0shortDecayLength);
      histos.fill(HIST("K0sGamma/BeforeSel/hV0InvMassWindow"), k0short.mK0Short() - o2::constants::physics::MassK0Short);
      histos.fill(HIST("K0sGamma/BeforeSel/h2dCompetingMassRej"), k0short.mLambda(), k0short.mK0Short());
      histos.fill(HIST("K0sGamma/BeforeSel/hPhotonMass"), k0short.mGamma());
      histos.fill(HIST("K0sGamma/BeforeSel/hPhotonZconv"), std::abs(k0short.z()));
      histos.fill(HIST("K0sGamma/BeforeSel/h2dArmenteros"), k0short.alpha(), k0short.qtarm());
      histos.fill(HIST("K0sGamma/BeforeSel/hPosTPCNsigmaPi"), posTrackExtraK0Short.tpcNSigmaPi());
      histos.fill(HIST("K0sGamma/BeforeSel/hNegTPCNsigmaPi"), negTrackExtraK0Short.tpcNSigmaPi());
      histos.fill(HIST("K0sGamma/BeforeSel/hPosTPCNsigmaEl"), posTrackExtraK0Short.tpcNSigmaEl());
      histos.fill(HIST("K0sGamma/BeforeSel/hNegTPCNsigmaEl"), negTrackExtraK0Short.tpcNSigmaEl());
      histos.fill(HIST("K0sGamma/BeforeSel/h2dPositiveITSvsTPCpts"), posTrackExtraK0Short.tpcCrossedRows(), posTrackExtraK0Short.itsNCls());
      histos.fill(HIST("K0sGamma/BeforeSel/h2dNegativeITSvsTPCpts"), negTrackExtraK0Short.tpcCrossedRows(), negTrackExtraK0Short.itsNCls());
    } else {
      // Candidates after K0s selections
      histos.fill(HIST("K0sGamma/K0s/hPosDCAToPV"), k0short.dcapostopv());
      histos.fill(HIST("K0sGamma/K0s/hNegDCAToPV"), k0short.dcanegtopv());
      histos.fill(HIST("K0sGamma/K0s/hDCAV0Daughters"), k0short.dcaV0daughters());
      histos.fill(HIST("K0sGamma/K0s/hDCAV0ToPV"), k0short.dcav0topv());
      histos.fill(HIST("K0sGamma/K0s/hV0PointingAngle"), k0short.v0cosPA());
      histos.fill(HIST("K0sGamma/K0s/hV0Radius"), k0short.v0radius());
      histos.fill(HIST("K0sGamma/K0s/hV0DecayLength"), k0shortDecayLength);
      histos.fill(HIST("K0sGamma/K0s/hV0InvMassWindow"), k0short.mK0Short() - o2::constants::physics::MassK0Short);
      histos.fill(HIST("K0sGamma/K0s/h2dCompetingMassRej"), k0short.mLambda(), k0short.mK0Short());
      histos.fill(HIST("K0sGamma/K0s/h2dArmenteros"), k0short.alpha(), k0short.qtarm());
      histos.fill(HIST("K0sGamma/K0s/hPosTPCNsigma"), posTrackExtraK0Short.tpcNSigmaPi());
      histos.fill(HIST("K0sGamma/K0s/hNegTPCNsigma"), negTrackExtraK0Short.tpcNSigmaPi());
      histos.fill(HIST("K0sGamma/K0s/h2dPositiveITSvsTPCpts"), posTrackExtraK0Short.tpcCrossedRows(), posTrackExtraK0Short.itsNCls());
      histos.fill(HIST("K0sGamma/K0s/h2dNegativeITSvsTPCpts"), negTrackExtraK0Short.tpcCrossedRows(), negTrackExtraK0Short.itsNCls());
      // Candidates after Gamma selections
      histos.fill(HIST("K0sGamma/Gamma/hPosDCAToPV"), gamma.dcapostopv());
      histos.fill(HIST("K0sGamma/Gamma/hNegDCAToPV"), gamma.dcanegtopv());
      histos.fill(HIST("K0sGamma/Gamma/hDCAV0Daughters"), gamma.dcaV0daughters());
      histos.fill(HIST("K0sGamma/Gamma/hDCAV0ToPV"), gamma.dcav0topv());
      histos.fill(HIST("K0sGamma/Gamma/hV0PointingAngle"), gamma.v0cosPA());
      histos.fill(HIST("K0sGamma/Gamma/hV0Radius"), gamma.v0radius());
      histos.fill(HIST("K0sGamma/Gamma/hPhotonMass"), gamma.mGamma());
      histos.fill(HIST("K0sGamma/Gamma/hPhotonZconv"), std::abs(gamma.z()));
      histos.fill(HIST("K0sGamma/Gamma/h2dArmenteros"), gamma.alpha(), gamma.qtarm());
      histos.fill(HIST("K0sGamma/Gamma/hPosTPCNsigma"), posTrackExtraGamma.tpcNSigmaEl());
      histos.fill(HIST("K0sGamma/Gamma/hNegTPCNsigma"), negTrackExtraGamma.tpcNSigmaEl());
      histos.fill(HIST("K0sGamma/Gamma/h2dPositiveITSvsTPCpts"), posTrackExtraGamma.tpcCrossedRows(), posTrackExtraGamma.itsNCls());
      histos.fill(HIST("K0sGamma/Gamma/h2dNegativeITSvsTPCpts"), negTrackExtraGamma.tpcCrossedRows(), negTrackExtraGamma.itsNCls());
    }
  }

  template <typename TCollision, typename TV0>
  void analyseV0PairCandidate(TCollision collision, TV0 k0short, TV0 gamma, float centrality, uint8_t gapSide)
  // fill information related to the resonance
  {
    float pt = RecoDecay::pt(k0short.px() + gamma.px(), k0short.py() + gamma.py());

    float invmass = RecoDecay::m(std::array{std::array{k0short.px(), k0short.py(), k0short.pz()}, std::array{gamma.px(), gamma.py(), gamma.pz()}}, std::array{o2::constants::physics::MassKaonNeutral, o2::constants::physics::MassGamma});

    float rapidity = RecoDecay::y(std::array{k0short.px() + gamma.px(), k0short.py() + gamma.py(), k0short.pz() + gamma.pz()}, invmass);

    // rapidity cut on the resonance
    if (!doMCAssociation && std::fabs(rapidity) > rapidityCut)
      return;

    // __________________________________________
    // main analysis
    if (doMCAssociation) {
      if constexpr (requires { k0short.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>(); }) { // check if MC information is available
        auto k0shortMC = k0short.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();
        auto gammaMC = gamma.template v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

        if (k0shortMC.pdgCodeMother() != gammaMC.pdgCodeMother()) {
          return;
        }

        float ptmc = RecoDecay::pt(k0shortMC.pxMC() + gammaMC.pxMC(), k0shortMC.pyMC() + gammaMC.pyMC());
        float rapiditymc = RecoDecay::y(std::array{k0shortMC.pxMC() + gammaMC.pxMC(), k0shortMC.pyMC() + gammaMC.pyMC(), k0shortMC.pzMC() + gammaMC.pzMC()}, pdgDB->Mass(k0shortMC.pdgCodeMother()));

        if (std::fabs(rapiditymc) > rapidityCut)
          return;

        if (k0shortMC.pdgCodeMother() == 313 && k0shortMC.pdgCodeMother() == gammaMC.pdgCodeMother()) { // 
          histos.fill(HIST("K0sGamma/h3dInvMassTrueK0Star892"), centrality, ptmc, invmass);
        }
      }
    }

    histos.fill(HIST("K0sGamma/h3dMassK0sGamma"), centrality, pt, invmass);
    if (!doPPAnalysis) { // in case of PbPb data
      if (gapSide == 0)
        histos.fill(HIST("K0sGamma/h3dMassK0sGammaSGA"), centrality, pt, invmass);
      else if (gapSide == 1)
        histos.fill(HIST("K0sGamma/h3dMassK0sGammaSGC"), centrality, pt, invmass);
      else if (gapSide == 2)
        histos.fill(HIST("K0sGamma/h3dMassK0sGammaDG"), centrality, pt, invmass);
      else
        histos.fill(HIST("K0sGamma/h3dMassK0sGammaHadronic"), centrality, pt, invmass);
    }
    fillQAplot(collision, k0short, gamma, 1);
  }

  // function to check that the k0short and gamma have different daughter tracks
  template <typename TV0>
  bool checkTrackIndices(TV0 k0short, TV0 gamma)
  {
    // check that positive track from k0short is different from daughter tracks of gamma
    if (k0short.posTrackExtraId() == gamma.posTrackExtraId() ||
        k0short.posTrackExtraId() == gamma.negTrackExtraId())
      return false;
    // check that negative track from k0short is different from daughter tracks of gamma
    if (k0short.negTrackExtraId() == gamma.posTrackExtraId() ||
        k0short.negTrackExtraId() == gamma.negTrackExtraId())
      return false;

    return true;
  }

  template <typename TCollision, typename TV0s>
  void buildV0V0Pairs(TCollision const& collision, TV0s const& fullV0s, std::vector<bool> selK0ShortIndices, std::vector<bool> selGammaIndices, float centrality, uint8_t gapSide)
  {
    // 1st loop over all v0s
    for (const auto& k0short : fullV0s) {
      // select only v0s matching Lambda selections
      if (!selK0ShortIndices[k0short.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
        continue;
      }

      // 2nd loop over all v0s
      for (const auto& gamma : fullV0s) {
        // select only v0s matching Anti-Lambda selections
        if (!selGammaIndices[gamma.globalIndex() - fullV0s.offset()]) { // local index needed due to collisions grouping
          continue;
        }

        // check we don't look at the same v0s/cascades
        if (k0short.globalIndex() == gamma.globalIndex()) {
          continue;
        }

        // check that the two hyperons have different daughter tracks
        if (!checkTrackIndices(k0short, gamma)) {
          continue;
        }

        // form V0 pairs and fill histograms
        analyseV0PairCandidate(collision, k0short, gamma, centrality, gapSide);
      } // end gamma loop
    } // end v0 loop

    return;
  }

  // ______________________________________________________
  // Real data processing - no MC subscription
  void processRealData(soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraStamps>::iterator const& collision, V0Candidates const& fullV0s, DauTracks const&)
  {
    // Fire up CCDB
    if (cfgSkimmedProcessing ||
        (mlConfigurations.useK0ShortScores && mlConfigurations.calculateK0ShortScores) ||
        (mlConfigurations.useLambdaScores && mlConfigurations.calculateLambdaScores) ||
        (mlConfigurations.useAntiLambdaScores && mlConfigurations.calculateAntiLambdaScores) ||
        (mlConfigurations.useGammaScores && mlConfigurations.calculateGammaScores)) {
      initCCDB(collision);
    }

    if (!isEventAccepted(collision, true)) {
      return;
    }

    if (cfgSkimmedProcessing) {
      zorro.isSelected(collision.globalBC()); /// Just let Zorro do the accounting
    }

    float centrality = -1;
    int selGapSide = -1; // only useful in case one wants to use this task in Pb-Pb UPC
    fillEventHistograms(collision, centrality, selGapSide);

    // __________________________________________
    // perform main analysis
    //
    std::vector<bool> selK0ShortIndices(fullV0s.size());
    std::vector<bool> selGammaIndices(fullV0s.size());
    for (const auto& v0 : fullV0s) {
      // fill QA plot before any selections
      fillQAplot(collision, v0, v0, 0);
      // identify whether the V0 matches the K0short and/or gamma selections
      analyseV0Candidate(v0, collision, v0.pt(), selK0ShortIndices, selGammaIndices, fullV0s.offset());
    } // end v0 loop

    // count the number of K0s and Gamma passing the selections
    int nK0Shorts = std::count(selK0ShortIndices.begin(), selK0ShortIndices.end(), true);
    int nGammas = std::count(selGammaIndices.begin(), selGammaIndices.end(), true);

    // fill the histograms with the number of reconstructed K0s/Gamma per collision
    histos.fill(HIST("K0sGamma/h2dNbrOfK0ShortVsCentrality"), centrality, nK0Shorts);
    histos.fill(HIST("K0sGamma/h2dNbrOfGammaVsCentrality"), centrality, nGammas);

    // Check the number of K0Short and Gamma
    // needs at least 1 of each
    if (nK0Shorts >= 1 && nGammas >= 1) { // consider K0S Gamma pairs
      buildV0V0Pairs(collision, fullV0s, selK0ShortIndices, selGammaIndices, centrality, selGapSide);
    }
  }

  // ______________________________________________________
  // Simulated processing (subscribes to MC information too)
  void processMonteCarlo(soa::Join<aod::StraCollisions, aod::StraCents, aod::StraEvSels, aod::StraStamps, aod::StraCollLabels>::iterator const& collision, V0MCCandidates const& fullV0s, DauTracks const&, aod::MotherMCParts const&, soa::Join<aod::StraMCCollisions, aod::StraMCCollMults> const& /*mccollisions*/, soa::Join<aod::V0MCCores, aod::V0MCCollRefs> const&)
  {
    // Fire up CCDB
    if (cfgSkimmedProcessing ||
        (mlConfigurations.useK0ShortScores && mlConfigurations.calculateK0ShortScores) ||
        (mlConfigurations.useLambdaScores && mlConfigurations.calculateLambdaScores) ||
        (mlConfigurations.useAntiLambdaScores && mlConfigurations.calculateAntiLambdaScores) ||
        (mlConfigurations.useGammaScores && mlConfigurations.calculateGammaScores)) {
      initCCDB(collision);
    }

    if (!isEventAccepted(collision, true)) {
      return;
    }

    if (cfgSkimmedProcessing) {
      zorro.isSelected(collision.globalBC()); /// Just let Zorro do the accounting
    }

    float centrality = -1;
    int selGapSide = -1; // only useful in case one wants to use this task in Pb-Pb UPC
    fillEventHistograms(collision, centrality, selGapSide);

    // __________________________________________
    // perform main analysis
    std::vector<bool> selK0ShortIndices(fullV0s.size());
    std::vector<bool> selGammaIndices(fullV0s.size());
    for (const auto& v0 : fullV0s) {
      if (!v0.has_v0MCCore())
        continue;

      auto v0MC = v0.v0MCCore_as<soa::Join<aod::V0MCCores, aod::V0MCCollRefs>>();

      float ptmc = RecoDecay::sqrtSumOfSquares(v0MC.pxPosMC() + v0MC.pxNegMC(), v0MC.pyPosMC() + v0MC.pyNegMC());

      // fill QA plot before any selections
      fillQAplot(collision, v0, v0, 0);
      // identify whether the V0 matches the K0short and/or gamma selections
      analyseV0Candidate(v0, collision, ptmc, selK0ShortIndices, selGammaIndices, fullV0s.offset());
    } // end v0 loop

      // count the number of K0s and Gamma passing the selections
    int nK0Shorts = std::count(selK0ShortIndices.begin(), selK0ShortIndices.end(), true);
    int nGammas = std::count(selGammaIndices.begin(), selGammaIndices.end(), true);

    // fill the histograms with the number of reconstructed K0s/Gamma per collision
    histos.fill(HIST("K0sGamma/h2dNbrOfK0ShortVsCentrality"), centrality, nK0Shorts);
    histos.fill(HIST("K0sGamma/h2dNbrOfGammaVsCentrality"), centrality, nGammas);

    // Check the number of K0Short and Gamma
    // needs at least 1 of each
    if (nK0Shorts >= 1 && nGammas >= 1) { // consider K0S Gamma pairs
      buildV0V0Pairs(collision, fullV0s, selK0ShortIndices, selGammaIndices, centrality, selGapSide);
    }
  }

  PROCESS_SWITCH(KstarToK0Gamma, processRealData, "process as if real data", true);
  PROCESS_SWITCH(KstarToK0Gamma, processMonteCarlo, "process as if MC", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<KstarToK0Gamma>(cfgc)};
}
