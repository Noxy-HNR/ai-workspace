# Entirely offline: tiny headers, scratch files, extracted functions and fake launchers.
$ErrorActionPreference='Stop'
$stackRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $stackRoot 'Tools\scripts\Build-LlamaArgs.ps1')
$script:Checks=0
function Assert-Check {param([bool]$Condition,[string]$Label) if(-not $Condition){throw "FAILED: $Label"};$script:Checks++}
function Assert-Throws {param([scriptblock]$Action,[string]$Label) $bad=$false;try{& $Action|Out-Null}catch{$bad=$true};Assert-Check $bad $Label}
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $stackRoot 'dashboard-server.ps1'),[ref]$tokens,[ref]$errors)
Assert-Check ($errors.Count -eq 0) 'dashboard parses'
$wanted=@('Get-GgufHeader','Get-LlamaModelAdmission','Get-ModelPath','Resolve-LoraPath','Get-ModelDefaults','Get-ModelsConfig','Get-ModelSettingsFor','Save-ModelsConfig','Save-ModelConfigOnly','Start-ModelTrial','Get-AvailableModels')
foreach($node in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$false)){
 if($node.Name -in $wanted){. ([scriptblock]::Create($node.Extent.Text))}
}
$optional=$ast.Find({param($n)$n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$OptionalModelKeys'},$false)
. ([scriptblock]::Create($optional.Extent.Text))
$reader=$ast.Find({param($n)$n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$GgufHeaderReader'},$false)
. ([scriptblock]::Create($reader.Extent.Text))
$scratchRoot=[IO.Path]::GetFullPath((Join-Path $stackRoot 'Scratch'))
$Root=Join-Path $scratchRoot ('draft-safety-'+[guid]::NewGuid().ToString('N'))
$ModelsDir=Join-Path $Root 'models';$drafts=Join-Path $ModelsDir 'drafts'
$ModelsConfigFile=Join-Path $Root 'models.json';$script:GgufHeaderCache=@{}
$script:ModelStorageConfigFile=Join-Path $Root 'model-storage.json'
New-Item -ItemType Directory -Path $drafts -Force|Out-Null
function Write-TinyGguf {
 param([string]$Path,[string]$Architecture)
 $stream=[IO.File]::Create($Path);$writer=New-Object IO.BinaryWriter($stream)
 try{
  $writer.Write([Text.Encoding]::ASCII.GetBytes('GGUF'));$writer.Write([uint32]3);$writer.Write([uint64]0);$writer.Write([uint64]2)
  foreach($entry in @(@('general.architecture',8,$Architecture),@("$Architecture.block_count",4,32))){
   $key=[Text.Encoding]::UTF8.GetBytes($entry[0]);$writer.Write([uint64]$key.Length);$writer.Write($key);$writer.Write([uint32]$entry[1])
   if($entry[1] -eq 8){$value=[Text.Encoding]::UTF8.GetBytes($entry[2]);$writer.Write([uint64]$value.Length);$writer.Write($value)}else{$writer.Write([uint32]$entry[2])}
  }
 }finally{$writer.Dispose();$stream.Dispose()}
}
try{
 $name='Qwen3.5-9B-DFlash-old-Q4_K_M.gguf';$path=Join-Path $drafts $name
 Write-TinyGguf $path 'dflash'
 Write-TinyGguf (Join-Path $ModelsDir 'target.gguf') 'qwen35'
 Write-TinyGguf (Join-Path $drafts 'ordinary.gguf') 'qwen35'
 Write-TinyGguf (Join-Path $drafts 'Bonsai-2-27B-DFlash2-Q8_0.gguf') 'dflash'
 Assert-Check ((Resolve-LlamaDraftFile -DraftFile $name -ModelsDir $ModelsDir) -eq $path) 'bare companion resolves only under drafts'
 Assert-Check ((Resolve-LlamaDraftFile -DraftFile ('drafts/'+$name) -ModelsDir $ModelsDir) -eq $path) 'relative drafts path resolves canonically'
 Assert-Check ((Resolve-LlamaDraftFile -DraftFile ('drafts\'+$name) -ModelsDir $ModelsDir) -eq $path) 'Windows relative drafts path resolves'
 Assert-Check ((Get-LlamaDraftArchitecture $path) -eq 'dflash') 'DFlash identity header recognized'
 foreach($empty in @($null,'')){Assert-Check ($null -eq (Resolve-LlamaDraftFile -DraftFile $empty -ModelsDir $ModelsDir)) 'unset draft field remains optional'}
 foreach($invalid in @($true,123,'../outside.gguf','drafts/../outside.gguf',$path,'C:\outside.gguf','\\server\draft.gguf','draft.gguf:stream','head.bin','missing.gguf','drafts/sub/head.gguf','bad"name.gguf',"bad'name.gguf",'head space.gguf','ordinary.gguf')){
  Assert-Throws {Resolve-LlamaDraftFile -DraftFile $invalid -ModelsDir $ModelsDir} "unsafe or non-DFlash companion refused: $invalid"
 }
 [IO.File]::WriteAllBytes((Join-Path $drafts 'broken.gguf'),[Text.Encoding]::ASCII.GetBytes('not a GGUF file despite its extension'))
 Assert-Throws {Resolve-LlamaDraftFile 'broken.gguf' $ModelsDir} 'nonGGUF bytes refused'
 $truncated=Join-Path $drafts 'truncated.gguf';[IO.File]::WriteAllBytes($truncated,([IO.File]::ReadAllBytes($path))[0..44])
 Assert-Throws {Resolve-LlamaDraftFile 'truncated.gguf' $ModelsDir} 'truncated identity string refused'
 $versionBytes=[IO.File]::ReadAllBytes($path);$versionBytes[4]=99
 [IO.File]::WriteAllBytes((Join-Path $drafts 'version.gguf'),$versionBytes)
 Assert-Throws {Resolve-LlamaDraftFile 'version.gguf' $ModelsDir} 'unsupported GGUF header version refused'
 $metadataBytes=[IO.File]::ReadAllBytes($path);for($i=16;$i -lt 24;$i++){$metadataBytes[$i]=0}
 [IO.File]::WriteAllBytes((Join-Path $drafts 'metadata.gguf'),$metadataBytes)
 Assert-Throws {Resolve-LlamaDraftFile 'metadata.gguf' $ModelsDir} 'missing GGUF metadata refused'
 $settings=Get-ModelDefaults;$settings.extraArgs='--spec-type draft-dflash --load-mode none --cache-ram 0';$settings.draftFile='drafts/'+$name
 $args=@(Build-LlamaArgs -ModelPath (Join-Path $ModelsDir 'target.gguf') -Settings $settings)
 Assert-Check ((@($args|Where-Object{$_ -eq '--model-draft'})).Count -eq 1) 'builder emits one companion switch'
 $position=[Array]::IndexOf($args,'--model-draft')
 Assert-Check ($args[$position+1] -eq $path) 'builder emits exact resolved companion path as one token'
 Assert-Check (($args -join ' ') -match '--load-mode none --cache-ram 0') 'mandatory RAM safety flags survive'
 foreach($unsafe in @('-md any.gguf','--model-draft any.gguf','--model-draft=any.gguf','--spec-draft-model any.gguf')){
  Assert-Throws {Get-LlamaSafeExtraArgs $unsafe} 'raw draft model arguments remain prohibited'
 }
 $settings.draftFile='ordinary.gguf';Assert-Throws {Build-LlamaArgs (Join-Path $ModelsDir 'target.gguf') $settings} 'ordinary-model companion never reaches launchargs'
 $settings.draftFile='drafts/'+$name
 @{activeModel='target.gguf';models=@{'target.gguf'=$settings}}|ConvertTo-Json -Depth 8|Set-Content $ModelsConfigFile -Encoding UTF8
 $saved=[pscustomobject]$settings
 Assert-Check (Save-ModelConfigOnly 'target.gguf' $saved).success 'typed draft field saved'
 Assert-Check ((Get-ModelsConfig).models.'target.gguf'.draftFile -eq ('drafts/'+$name)) 'saved draft field preserved'
 $saved.PSObject.Properties.Remove('draftFile')
 Assert-Check ((Save-ModelConfigOnly 'target.gguf' $saved).success -and (Get-ModelsConfig).models.'target.gguf'.draftFile -eq ('drafts/'+$name)) 'ordinary form save preserves omitted draft'
 $before=(Get-FileHash $ModelsConfigFile).Hash
 $saved|Add-Member -NotePropertyName draftFile -NotePropertyValue 'ordinary.gguf'
 Assert-Check (-not (Save-ModelConfigOnly 'target.gguf' $saved).success -and (Get-FileHash $ModelsConfigFile).Hash -eq $before) 'invalid draft cannot overwrite settings'
 $saved.draftFile=''
 Assert-Check ((Save-ModelConfigOnly 'target.gguf' $saved).success -and -not (Get-ModelsConfig).models.'target.gguf'.draftFile) 'explicit empty field clears saved companion'
 $saved.draftFile='drafts/'+$name
 Assert-Check (Save-ModelConfigOnly 'target.gguf' $saved).success 'valid field restored'
 function Get-LlamaPhysicalMemory {return @{availableBytes=8GB}}
 function Get-LlamaMemoryAdmission {param($ModelBytes,$BlockCount,$GpuLayers,$AvailableBytes,$AdapterBytes,$MinimumAvailableBytes,$CpuMoeLayers) return @{allowed=$true;draftAndAdapterBytes=$AdapterBytes}}
 $admission=Get-LlamaModelAdmission 'target.gguf' $settings
 Assert-Check ($admission.draftAndAdapterBytes -eq (Get-Item -LiteralPath $path).Length) 'full draft file charged as host companion allocation'
 [IO.File]::WriteAllBytes((Join-Path $ModelsDir 'adapter.gguf'),(New-Object byte[] 123))
 $settings.loraFile='adapter.gguf'
 Assert-Check ((Get-LlamaModelAdmission 'target.gguf' $settings).draftAndAdapterBytes -eq ((Get-Item -LiteralPath $path).Length+123)) 'draft and LoRA host allocation are both fully charged'
 & {
  function Resolve-LlamaDraftFile {param($DraftFile,$ModelsDir)return $path}
  function Get-Item {param($LiteralPath,$ErrorAction) if($LiteralPath -eq $path){return [pscustomobject]@{Length=[long]2.5GB}};return Microsoft.PowerShell.Management\Get-Item -LiteralPath $LiteralPath -ErrorAction Stop}
  Assert-Check ((Get-LlamaModelAdmission 'target.gguf' $settings).draftAndAdapterBytes -eq ([long]2.5GB+123)) 'large companion size is preserved in 64-bit host charge'
 }
 $settings.Remove('loraFile')
 $script:Starts=0
 function Start-LlamaNative {param($FileName,$Settings) $script:Starts++;$script:TrialSettings=$Settings;$built=@(Build-LlamaArgs (Join-Path $ModelsDir $FileName) $Settings);return @{success=$true}}
 $before=(Get-FileHash $ModelsConfigFile).Hash
 $trial=Start-ModelTrial 'target.gguf' ([pscustomobject]@{draftFile='Bonsai-2-27B-DFlash2-Q8_0.gguf';contextSize=8192})
 Assert-Check ($trial.success -and $script:TrialSettings.draftFile -eq 'Bonsai-2-27B-DFlash2-Q8_0.gguf') 'persistfalse typed companion override reaches builder'
 Assert-Check ((Get-FileHash $ModelsConfigFile).Hash -eq $before) 'trial does not save companion or settings'
 Assert-Check (-not (Start-ModelTrial 'target.gguf' ([pscustomobject]@{extraArgs='-md any.gguf'})).success) 'trial cannot bypass companion field through extraArgs'
 $listed=Get-AvailableModels
 Assert-Check ($name -notin $listed.models.filename -and 'Bonsai-2-27B-DFlash2-Q8_0.gguf' -notin $listed.models.filename) 'nested companions never become chat roster entries'
 $held=Join-Path $ModelsDir 'held-drafts';Move-Item -LiteralPath $drafts -Destination $held
 $outside=Join-Path $Root 'outside-drafts';New-Item -ItemType Directory -Path $outside|Out-Null;Write-TinyGguf (Join-Path $outside $name) 'dflash'
 New-Item -ItemType Junction -Path $drafts -Target $outside|Out-Null
 try{Assert-Throws {Resolve-LlamaDraftFile $name $ModelsDir} 'junction draft folder refused'}finally{
  $junctionPath=[IO.Path]::GetFullPath($drafts)
  if(-not $junctionPath.StartsWith([IO.Path]::GetFullPath($Root)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Unsafe fixture junction cleanup path'}
  [IO.Directory]::Delete($junctionPath);Move-Item -LiteralPath $held -Destination $drafts
 }
 Write-Output "PASS: $script:Checks offline draft assertions; no model processes launched."
}finally{
 $resolved=[IO.Path]::GetFullPath($Root)
 if($resolved.StartsWith($scratchRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){Remove-Item -LiteralPath $resolved -Recurse -Force}
}
